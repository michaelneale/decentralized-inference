//! Filesystem facts the L3 disk tier has to know before it writes.
//!
//! The tier promises a minimum-free-space reserve, owner-only permissions, a
//! refusal to run on network filesystems, and a recency signal cheap enough to
//! update on every cache hit. None of that is in `std`, so the platform calls
//! live here and the store stays free of `unsafe`.

use std::{
    fs,
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context, Result, bail};

/// Bytes an unprivileged writer can still add to the filesystem holding
/// `path`. This is `f_bavail`, not `f_bfree`: the reserve check must not spend
/// blocks only root can allocate.
pub fn available_bytes(path: &Path) -> Result<u64> {
    fs2::available_space(path)
        .with_context(|| format!("failed to stat available space for {}", path.display()))
}

/// Whether `path` sits on a filesystem the tier refuses to manage. Network
/// filesystems break the atomic-rename and locking assumptions the store is
/// built on, so §10.10 rejects them where they are reliably detectable.
pub fn is_network_filesystem(path: &Path) -> Result<bool> {
    #[cfg(windows)]
    {
        return windows::is_network_filesystem(path);
    }
    #[cfg(not(windows))]
    let name = filesystem_type_name(path)?;
    #[cfg(not(windows))]
    {
        Ok(matches!(
            name.as_str(),
            "nfs" | "smbfs" | "afpfs" | "webdav" | "ftp" | "cifs" | "fuse" | "fuse.sshfs"
        ))
    }
}

/// The filesystem type as the kernel names it, for status reporting.
pub fn filesystem_type_name(path: &Path) -> Result<String> {
    #[cfg(windows)]
    {
        return windows::filesystem_type_name(path);
    }
    #[cfg(target_os = "macos")]
    {
        let stat = statfs(path)?;
        let raw = stat.f_fstypename;
        let bytes: Vec<u8> = raw
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    #[cfg(all(not(target_os = "macos"), not(windows)))]
    {
        // Linux reports a magic number rather than a name. Only the values the
        // tier actually refuses are worth naming; anything else is local
        // enough to manage.
        let stat = statfs(path)?;
        Ok(match stat.f_type {
            0x6969 => "nfs".to_string(),
            0xFF53_4D42 => "cifs".to_string(),
            // FUSE_SUPER_MAGIC. statfs cannot name the subtype, so every FUSE
            // mount reports as plain "fuse": sshfs and a local fuse filesystem
            // are indistinguishable here. The tier refuses the whole class
            // rather than guess, which is the conservative reading of §10.10.
            0x6573_5546 => "fuse".to_string(),
            other => format!("0x{other:x}"),
        })
    }
}

/// Mark an entry as used now, so eviction can order by last use rather than
/// last write. One `utimensat` per cache hit is the bounded metadata update
/// §13.4 allows; it writes no payload bytes and allocates no blocks.
pub fn touch(path: &Path) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_times(fs::FileTimes::new().set_modified(SystemTime::now())))
        .with_context(|| format!("failed to touch {}", path.display()))
}

/// Publish a fully written temporary file at `destination`, replacing an
/// existing entry when necessary. Windows `rename` does not replace an
/// existing file, so use the platform primitive with explicit replacement.
pub fn replace_file(temp: &Path, destination: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        return windows::replace_file(temp, destination);
    }
    #[cfg(not(windows))]
    {
        fs::rename(temp, destination)
            .with_context(|| format!("failed to publish {}", destination.display()))
    }
}

/// Restrict a cache directory to its owner. The first release has no at-rest
/// encryption, so local account permissions are the only confidentiality the
/// tier offers and it must actually apply them.
pub fn restrict_to_owner(path: &Path, mode: u32) -> Result<()> {
    #[cfg(windows)]
    {
        let _ = mode;
        return windows::restrict_to_owner(path);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::Permissions::from_mode(mode);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("failed to restrict permissions on {}", path.display()))
    }
}

/// Refuse a path that reaches the store through a symlink. Following one would
/// let anything with write access to the parent redirect committed cache
/// bytes outside the managed root, past every budget and reserve check.
pub fn refuse_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if is_link_or_reparse_point(&metadata) => {
            bail!(
                "{} is a symlink; the cache refuses to traverse it",
                path.display()
            )
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to stat {}", path.display())),
    }
}

/// Create an absolute directory tree without traversing an untrusted link.
/// Existing ancestors are inspected before any missing component is created,
/// closing the gap where `create_dir_all` could follow a redirected parent.
pub fn create_dir_all_without_links(path: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        if matches!(component, Component::Prefix(_)) {
            current.push(component.as_os_str());
            continue;
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if is_link_or_reparse_point(&metadata) => {
                if current != path && is_trusted_platform_directory_link(&current) {
                    continue;
                }
                bail!(
                    "{} is a symlink; the cache refuses to traverse it",
                    current.display()
                );
            }
            Ok(metadata) if !metadata.is_dir() => {
                bail!("{} is not a directory", current.display());
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&current).with_context(|| {
                            format!("failed to verify cache directory {}", current.display())
                        })?;
                        if is_link_or_reparse_point(&metadata) || !metadata.is_dir() {
                            bail!(
                                "{} appeared during creation but is not a safe directory",
                                current.display()
                            );
                        }
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("failed to create cache directory {}", current.display())
                        });
                    }
                }
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to stat {}", current.display()));
            }
        }
    }
    refuse_symlink(path)
}

fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        return metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0;
    }
    #[cfg(not(windows))]
    false
}

fn is_trusted_platform_directory_link(_path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(_path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// Refuse a symlink anywhere in the portion of `path` below `root`.
///
/// The system prefix above the cache root is not ours to police: on macOS
/// `/var` is itself a symlink to `/private/var`, so refusing every symlinked
/// ancestor would reject the default temp and cache locations. What must hold
/// is that nothing the store creates under its own resolved root redirects
/// bytes outside it.
pub fn refuse_symlinked_descendant(root: &Path, path: &Path) -> Result<()> {
    let Ok(relative) = path.strip_prefix(root) else {
        bail!(
            "{} is not inside the cache root {}",
            path.display(),
            root.display()
        );
    };
    let mut walked = root.to_path_buf();
    for component in relative.components() {
        walked.push(component);
        refuse_symlink(&walked)?;
    }
    Ok(())
}

#[cfg(not(windows))]
fn c_path(path: &Path) -> Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("path {} contains an interior NUL", path.display()))
}

#[cfg(not(windows))]
fn statfs(path: &Path) -> Result<libc::statfs> {
    let c_path = c_path(path)?;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: as `statvfs` above.
    let status = unsafe { libc::statfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if status != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to stat filesystem for {}", path.display()));
    }
    // SAFETY: statfs returned 0, so `stat` is initialized.
    Ok(unsafe { stat.assume_init() })
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::ffi::c_void;
    use std::mem::{align_of, size_of};
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::Security::Authorization::{SE_FILE_OBJECT, SetNamedSecurityInfoW};
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_REVISION, AddAccessAllowedAceEx, CONTAINER_INHERIT_ACE,
        DACL_SECURITY_INFORMATION, GetLengthSid, GetTokenInformation, InitializeAcl,
        OBJECT_INHERIT_ACE, OWNER_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSID,
        TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ALL_ACCESS, GetDriveTypeW, GetVolumeInformationW, GetVolumePathNameW,
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::Win32::System::WindowsProgramming::DRIVE_REMOTE;

    pub(super) fn is_network_filesystem(path: &Path) -> Result<bool> {
        let volume = volume_root(path)?;
        Ok(unsafe { GetDriveTypeW(volume.as_ptr()) } == DRIVE_REMOTE)
    }

    pub(super) fn filesystem_type_name(path: &Path) -> Result<String> {
        let volume = volume_root(path)?;
        let mut filesystem = [0_u16; 64];
        let ok = unsafe {
            GetVolumeInformationW(
                volume.as_ptr(),
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                null_mut(),
                filesystem.as_mut_ptr(),
                filesystem.len() as u32,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to inspect filesystem for {}", path.display()));
        }
        let length = filesystem
            .iter()
            .position(|unit| *unit == 0)
            .unwrap_or(filesystem.len());
        Ok(String::from_utf16_lossy(&filesystem[..length]).to_ascii_lowercase())
    }

    pub(super) fn replace_file(temp: &Path, destination: &Path) -> Result<()> {
        let temp_wide = to_wide(temp);
        let destination_wide = to_wide(destination);
        let result = unsafe {
            MoveFileExW(
                temp_wide.as_ptr(),
                destination_wide.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        };
        if result == 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to publish {}", destination.display()));
        }
        Ok(())
    }

    fn volume_root(path: &Path) -> Result<Vec<u16>> {
        let input = to_wide(path);
        let mut volume = vec![0_u16; 260];
        let ok =
            unsafe { GetVolumePathNameW(input.as_ptr(), volume.as_mut_ptr(), volume.len() as u32) };
        if ok == 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to resolve volume for {}", path.display()));
        }
        Ok(volume)
    }

    pub(super) fn restrict_to_owner(path: &Path) -> Result<()> {
        with_current_user_sid(|sid| {
            let acl_bytes = size_of::<ACL>() + size_of::<ACCESS_ALLOWED_ACE>() - size_of::<u32>()
                + unsafe { GetLengthSid(sid) as usize };
            let words = acl_bytes.div_ceil(size_of::<u64>());
            let mut acl_storage = vec![0_u64; words];
            let acl = acl_storage.as_mut_ptr().cast::<ACL>();
            let metadata = fs::metadata(path)?;
            let ace_flags = if metadata.is_dir() {
                OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
            } else {
                0
            };
            unsafe {
                if InitializeAcl(acl, acl_bytes as u32, ACL_REVISION) == 0
                    || AddAccessAllowedAceEx(acl, ACL_REVISION, ace_flags, FILE_ALL_ACCESS, sid)
                        == 0
                {
                    return Err(std::io::Error::last_os_error().into());
                }
            }
            let mut wide = to_wide(path);
            let result = unsafe {
                SetNamedSecurityInfoW(
                    wide.as_mut_ptr(),
                    SE_FILE_OBJECT,
                    OWNER_SECURITY_INFORMATION
                        | DACL_SECURITY_INFORMATION
                        | PROTECTED_DACL_SECURITY_INFORMATION,
                    sid,
                    null_mut(),
                    acl,
                    null(),
                )
            };
            if result != 0 {
                bail!("Windows ACL update failed with error {result}");
            }
            Ok(())
        })
        .with_context(|| format!("failed to restrict permissions on {}", path.display()))
    }

    fn with_current_user_sid<T>(f: impl FnOnce(PSID) -> Result<T>) -> Result<T> {
        let mut token = null_mut();
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let _token = Handle(token);
        let mut bytes = 0_u32;
        unsafe {
            let _ = GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes);
        }
        if bytes == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let words = (bytes as usize).div_ceil(align_of::<usize>());
        let mut buffer = vec![0_usize; words];
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast::<c_void>(),
                bytes,
                &mut bytes,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        let token_user = unsafe { &*buffer.as_ptr().cast::<TOKEN_USER>() };
        f(token_user.User.Sid)
    }

    fn to_wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    struct Handle(windows_sys::Win32::Foundation::HANDLE);

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_bytes_reports_a_plausible_figure() {
        let available = available_bytes(Path::new(".")).expect("stat working directory");
        assert!(available > 0, "working directory reports no free space");
    }

    #[test]
    fn touch_moves_modification_time_forward() {
        let directory =
            std::env::temp_dir().join(format!("skippy-fsinfo-touch-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create temp dir");
        let path = directory.join("entry");
        fs::write(&path, b"entry").expect("write entry");
        let before = fs::metadata(&path)
            .expect("stat")
            .modified()
            .expect("mtime");
        std::thread::sleep(std::time::Duration::from_millis(20));
        touch(&path).expect("touch entry");
        let after = fs::metadata(&path)
            .expect("stat")
            .modified()
            .expect("mtime");
        assert!(after > before, "touch did not move the modification time");
        fs::remove_dir_all(&directory).ok();
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_refused() {
        let directory =
            std::env::temp_dir().join(format!("skippy-fsinfo-symlink-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create temp dir");
        let target = directory.join("target");
        let link = directory.join("link");
        fs::write(&target, b"target").expect("write target");
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).expect("create symlink");
        assert!(refuse_symlink(&link).is_err(), "symlink was accepted");
        assert!(refuse_symlink(&target).is_ok(), "regular file was refused");
        assert!(
            refuse_symlink(&directory.join("absent")).is_ok(),
            "absent path was refused"
        );
        fs::remove_dir_all(&directory).ok();
    }
}
