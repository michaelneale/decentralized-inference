//! Producer-facing runtime-event contracts.

macro_rules! event_family {
    ($name:ident { $($variant:ident => $id:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $id),+
                }
            }
        }
    };
}

mod carrier;
mod delivery;
mod fact_data;
mod facts;
mod families;
mod identity;
mod ingress;
mod native;

pub use carrier::*;
pub use delivery::*;
pub use fact_data::*;
pub use facts::*;
pub use families::*;
pub use identity::*;
pub use ingress::*;
pub use native::*;

#[cfg(test)]
mod tests;
