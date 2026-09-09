use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComponentKind {
    ResidentKv,
    KvRecurrent,
}

#[derive(Debug, Clone)]
struct ComponentEntry<T> {
    value: T,
    logical_bytes: u64,
    last_used: u64,
    active_refs: u32,
}

#[derive(Debug)]
struct RadixComponents<R, E> {
    resident: Option<ComponentEntry<R>>,
    recurrent: Option<ComponentEntry<E>>,
}

impl<R, E> Default for RadixComponents<R, E> {
    fn default() -> Self {
        Self {
            resident: None,
            recurrent: None,
        }
    }
}

#[derive(Debug)]
struct RadixNode<R, E> {
    edge: Vec<i32>,
    components: RadixComponents<R, E>,
    children: BTreeMap<i32, RadixNode<R, E>>,
}

impl<R, E> RadixNode<R, E> {
    fn root() -> Self {
        Self::new(Vec::new())
    }

    fn new(edge: Vec<i32>) -> Self {
        Self {
            edge,
            components: RadixComponents::default(),
            children: BTreeMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.components.resident.is_none()
            && self.components.recurrent.is_none()
            && self.children.is_empty()
    }
}

/// One cache hit from the unified token radix tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixMatch<T> {
    /// Number of query tokens that can be restored from this entry.
    pub matched_tokens: usize,
    /// Full token path owning the stored payload. For resident KV this can be
    /// longer than `matched_tokens`: native sequence copy can restore only the
    /// common prefix from a longer cached sequence.
    pub stored_tokens: Vec<i32>,
    pub logical_bytes: u64,
    pub active_refs: u32,
    pub value: T,
}

/// One component payload removed by radix-aware eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixEviction<T> {
    pub namespace: String,
    pub tokens: Vec<i32>,
    pub logical_bytes: u64,
    pub value: T,
}

/// One unreferenced component that may be released by an external capacity
/// planner. Selecting a candidate does not mutate recency or the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixEvictionCandidate<T> {
    pub namespace: String,
    pub tokens: Vec<i32>,
    pub logical_bytes: u64,
    pub last_used: u64,
    pub value: T,
}

/// Aggregate metadata for the logical radix topology and its payloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnifiedRadixCacheStats {
    pub namespaces: usize,
    pub nodes: usize,
    pub token_edges: usize,
    pub splits: u64,
    pub resident_entries: usize,
    pub resident_tokens: u64,
    pub resident_logical_bytes: u64,
    pub resident_active_refs: u64,
    pub resident_pinned_tokens: u64,
    pub recurrent_entries: usize,
    pub recurrent_logical_bytes: u64,
    pub recurrent_active_refs: u64,
    pub resident_evictions: u64,
    pub recurrent_evictions: u64,
}

/// A compressed token radix tree shared by resident-KV and recurrent-state
/// cache components.
///
/// The tree owns exact prefix matching, node splitting, recency, reference
/// protection, pruning, and component-aware eviction. Payload mutation stays
/// outside this type: callers remain responsible for native sequence copy/drop
/// and exact-state import/export.
#[derive(Debug)]
pub struct UnifiedRadixCache<R, E> {
    clock: u64,
    splits: u64,
    resident_evictions: u64,
    recurrent_evictions: u64,
    roots: HashMap<String, RadixNode<R, E>>,
}

impl<R, E> Default for UnifiedRadixCache<R, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R, E> UnifiedRadixCache<R, E> {
    pub fn new() -> Self {
        Self {
            clock: 0,
            splits: 0,
            resident_evictions: 0,
            recurrent_evictions: 0,
            roots: HashMap::new(),
        }
    }

    pub fn insert_resident(
        &mut self,
        namespace: impl Into<String>,
        tokens: &[i32],
        logical_bytes: u64,
        value: R,
    ) -> Result<Option<R>> {
        validate_tokens(tokens)?;
        let namespace = namespace.into();
        if self
            .roots
            .get(&namespace)
            .and_then(|root| node_at(root, tokens))
            .and_then(|node| node.components.resident.as_ref())
            .is_some_and(|entry| entry.active_refs > 0)
        {
            bail!("cannot replace an active resident radix entry");
        }
        self.clock = self.clock.saturating_add(1);
        let last_used = self.clock;
        let node = self.ensure_node(namespace, tokens);
        Ok(node
            .components
            .resident
            .replace(ComponentEntry {
                value,
                logical_bytes,
                last_used,
                active_refs: 0,
            })
            .map(|entry| entry.value))
    }

    /// Insert a resident component without replacing an existing native
    /// backing sequence. Returns the rejected value when the key is occupied.
    pub fn insert_resident_if_vacant(
        &mut self,
        namespace: impl Into<String>,
        tokens: &[i32],
        logical_bytes: u64,
        value: R,
    ) -> Result<Option<R>> {
        validate_tokens(tokens)?;
        let namespace = namespace.into();
        if self
            .roots
            .get(&namespace)
            .and_then(|root| node_at(root, tokens))
            .and_then(|node| node.components.resident.as_ref())
            .is_some()
        {
            return Ok(Some(value));
        }
        self.clock = self.clock.saturating_add(1);
        let last_used = self.clock;
        let node = self.ensure_node(namespace, tokens);
        debug_assert!(node.components.resident.is_none());
        node.components.resident = Some(ComponentEntry {
            value,
            logical_bytes,
            last_used,
            active_refs: 0,
        });
        Ok(None)
    }

    pub fn insert_recurrent(
        &mut self,
        namespace: impl Into<String>,
        tokens: &[i32],
        logical_bytes: u64,
        value: E,
    ) -> Result<Option<E>> {
        validate_tokens(tokens)?;
        let namespace = namespace.into();
        if self
            .roots
            .get(&namespace)
            .and_then(|root| node_at(root, tokens))
            .and_then(|node| node.components.recurrent.as_ref())
            .is_some_and(|entry| entry.active_refs > 0)
        {
            bail!("cannot replace an active recurrent radix entry");
        }
        self.clock = self.clock.saturating_add(1);
        let last_used = self.clock;
        let node = self.ensure_node(namespace, tokens);
        Ok(node
            .components
            .recurrent
            .replace(ComponentEntry {
                value,
                logical_bytes,
                last_used,
                active_refs: 0,
            })
            .map(|entry| entry.value))
    }

    pub fn remove_resident(&mut self, namespace: &str, tokens: &[i32]) -> Option<R> {
        let root = self.roots.get_mut(namespace)?;
        let entry = node_at_mut(root, tokens)?.components.resident.take()?;
        normalize_root(root);
        if root.is_empty() {
            self.roots.remove(namespace);
        }
        Some(entry.value)
    }

    pub fn remove_recurrent(&mut self, namespace: &str, tokens: &[i32]) -> Option<E> {
        let root = self.roots.get_mut(namespace)?;
        let entry = node_at_mut(root, tokens)?.components.recurrent.take()?;
        normalize_root(root);
        if root.is_empty() {
            self.roots.remove(namespace);
        }
        Some(entry.value)
    }

    /// Remove an exact recurrent entry only when its value still matches the
    /// caller's predicate. The value check and removal happen under the same
    /// cache lock, so a failed restore can quarantine the entry it acquired
    /// without accidentally deleting a replacement inserted concurrently.
    pub fn remove_recurrent_if(
        &mut self,
        namespace: &str,
        tokens: &[i32],
        predicate: impl FnOnce(&E) -> bool,
    ) -> Option<E> {
        let should_remove = self
            .roots
            .get(namespace)
            .and_then(|root| node_at(root, tokens))
            .and_then(|node| node.components.recurrent.as_ref())
            .is_some_and(|entry| entry.active_refs == 0 && predicate(&entry.value));
        should_remove
            .then(|| self.remove_recurrent(namespace, tokens))
            .flatten()
    }

    pub fn remove_resident_where(
        &mut self,
        mut predicate: impl FnMut(&R) -> bool,
    ) -> Option<RadixEviction<R>> {
        let (namespace, tokens, logical_bytes) =
            self.roots.iter().find_map(|(namespace, root)| {
                find_resident_path(root, &mut Vec::new(), &mut predicate)
                    .map(|(tokens, logical_bytes)| (namespace.clone(), tokens, logical_bytes))
            })?;
        let value = self.remove_resident(&namespace, &tokens)?;
        Some(RadixEviction {
            namespace,
            tokens,
            logical_bytes,
            value,
        })
    }

    pub fn remove_recurrent_where(
        &mut self,
        mut predicate: impl FnMut(&E) -> bool,
    ) -> Option<RadixEviction<E>> {
        let (namespace, tokens, logical_bytes) =
            self.roots.iter().find_map(|(namespace, root)| {
                find_recurrent_path(root, &mut Vec::new(), &mut predicate)
                    .map(|(tokens, logical_bytes)| (namespace.clone(), tokens, logical_bytes))
            })?;
        let value = self.remove_recurrent(&namespace, &tokens)?;
        Some(RadixEviction {
            namespace,
            tokens,
            logical_bytes,
            value,
        })
    }

    pub fn evict_lru_resident(&mut self) -> Option<RadixEviction<R>> {
        let victim = self.lru_victim(ComponentKind::ResidentKv)?;
        self.evict_resident_candidate(&victim.namespace, &victim.tokens)
    }

    /// Remove one exact unreferenced resident candidate selected by an
    /// external policy. Active references fail closed and remain resident.
    pub fn evict_resident_candidate(
        &mut self,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<RadixEviction<R>> {
        let entry = node_at(self.roots.get(namespace)?, tokens)?
            .components
            .resident
            .as_ref()?;
        if entry.active_refs > 0 {
            return None;
        }
        let logical_bytes = entry.logical_bytes;
        let value = self.remove_resident(namespace, tokens)?;
        self.resident_evictions = self.resident_evictions.saturating_add(1);
        Some(RadixEviction {
            namespace: namespace.to_string(),
            tokens: tokens.to_vec(),
            logical_bytes,
            value,
        })
    }

    pub fn evict_lru_recurrent(&mut self) -> Option<RadixEviction<E>> {
        let victim = self.lru_victim(ComponentKind::KvRecurrent)?;
        let value = self.remove_recurrent(&victim.namespace, &victim.tokens)?;
        self.recurrent_evictions = self.recurrent_evictions.saturating_add(1);
        Some(RadixEviction {
            namespace: victim.namespace,
            tokens: victim.tokens,
            logical_bytes: victim.logical_bytes,
            value,
        })
    }

    pub fn stats(&self) -> UnifiedRadixCacheStats {
        let mut stats = UnifiedRadixCacheStats {
            namespaces: self.roots.len(),
            splits: self.splits,
            resident_evictions: self.resident_evictions,
            recurrent_evictions: self.recurrent_evictions,
            ..UnifiedRadixCacheStats::default()
        };
        for root in self.roots.values() {
            accumulate_stats(root, true, 0, &mut stats);
        }
        stats
    }

    /// Monotonic cache mutation/recency epoch used to identify stale
    /// scheduler observations. Peeks deliberately do not advance it.
    pub fn epoch(&self) -> u64 {
        self.clock
    }

    fn ensure_node(&mut self, namespace: String, tokens: &[i32]) -> &mut RadixNode<R, E> {
        let root = self.roots.entry(namespace).or_insert_with(RadixNode::root);
        ensure_node(root, tokens, &mut self.splits)
    }

    fn lru_victim(&self, component: ComponentKind) -> Option<Victim> {
        let mut victim = None;
        for (namespace, root) in &self.roots {
            collect_lru_victim(namespace, root, component, &mut Vec::new(), &mut victim);
        }
        victim
    }
}

impl<R: Clone, E> UnifiedRadixCache<R, E> {
    /// Snapshot all unreferenced resident entries for capacity planning.
    pub fn resident_eviction_candidates(&self) -> Vec<RadixEvictionCandidate<R>> {
        let mut candidates = Vec::new();
        for (namespace, root) in &self.roots {
            collect_resident_eviction_candidates(namespace, root, &mut Vec::new(), &mut candidates);
        }
        candidates
    }

    /// Find the resident prefix without changing recency or active references.
    /// Scheduler scans must not make merely considered entries look hot.
    pub fn peek_resident(&self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<R>> {
        let root = self.roots.get(namespace)?;
        let (matched_tokens, stored_tokens) = resident_backing_prefix(root, tokens)?;
        let entry = node_at(root, &stored_tokens)?
            .components
            .resident
            .as_ref()?;
        Some(RadixMatch {
            matched_tokens,
            stored_tokens,
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn resident_exact(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<R>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let entry = node_at_mut(root, tokens)?.components.resident.as_mut()?;
        entry.last_used = self.clock;
        Some(RadixMatch {
            matched_tokens: tokens.len(),
            stored_tokens: tokens.to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn lru_resident_candidate(&self) -> Option<RadixEviction<R>> {
        let victim = self.lru_victim(ComponentKind::ResidentKv)?;
        let root = self.roots.get(&victim.namespace)?;
        let value = node_at(root, &victim.tokens)?
            .components
            .resident
            .as_ref()?
            .value
            .clone();
        Some(RadixEviction {
            namespace: victim.namespace,
            tokens: victim.tokens,
            logical_bytes: victim.logical_bytes,
            value,
        })
    }

    pub fn lookup_resident(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<R>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let (matched_tokens, stored_tokens) = resident_backing_prefix(root, tokens)?;
        let entry = node_at_mut(root, &stored_tokens)?
            .components
            .resident
            .as_mut()?;
        entry.last_used = self.clock;
        Some(RadixMatch {
            matched_tokens,
            stored_tokens,
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn acquire_resident(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<R>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let (matched_tokens, stored_tokens) = resident_backing_prefix(root, tokens)?;
        let entry = node_at_mut(root, &stored_tokens)?
            .components
            .resident
            .as_mut()?;
        entry.last_used = self.clock;
        entry.active_refs = entry.active_refs.saturating_add(1);
        Some(RadixMatch {
            matched_tokens,
            stored_tokens,
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn release_resident(&mut self, namespace: &str, tokens: &[i32]) -> bool {
        let Some(root) = self.roots.get_mut(namespace) else {
            return false;
        };
        let Some(entry) =
            node_at_mut(root, tokens).and_then(|node| node.components.resident.as_mut())
        else {
            return false;
        };
        if entry.active_refs == 0 {
            return false;
        }
        entry.active_refs -= 1;
        self.clock = self.clock.saturating_add(1);
        entry.last_used = self.clock;
        true
    }

    pub fn release_resident_where(&mut self, mut predicate: impl FnMut(&R) -> bool) -> bool {
        let Some((namespace, tokens)) = self.roots.iter().find_map(|(namespace, root)| {
            find_resident_path(root, &mut Vec::new(), &mut predicate)
                .map(|(tokens, _)| (namespace.clone(), tokens))
        }) else {
            return false;
        };
        self.release_resident(&namespace, &tokens)
    }
}

impl<R, E: Clone> UnifiedRadixCache<R, E> {
    /// Find the recurrent prefix without changing recency or active references.
    /// Scheduler scans must not make merely considered entries look hot.
    pub fn peek_recurrent(&self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<E>> {
        let root = self.roots.get(namespace)?;
        let matched_tokens = longest_component_prefix(root, tokens, ComponentKind::KvRecurrent)?;
        let entry = node_at(root, &tokens[..matched_tokens])?
            .components
            .recurrent
            .as_ref()?;
        Some(RadixMatch {
            matched_tokens,
            stored_tokens: tokens[..matched_tokens].to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn recurrent_exact(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<E>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let entry = node_at_mut(root, tokens)?.components.recurrent.as_mut()?;
        entry.last_used = self.clock;
        Some(RadixMatch {
            matched_tokens: tokens.len(),
            stored_tokens: tokens.to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn lru_recurrent_candidate(&self) -> Option<RadixEviction<E>> {
        let victim = self.lru_victim(ComponentKind::KvRecurrent)?;
        let root = self.roots.get(&victim.namespace)?;
        let value = node_at(root, &victim.tokens)?
            .components
            .recurrent
            .as_ref()?
            .value
            .clone();
        Some(RadixEviction {
            namespace: victim.namespace,
            tokens: victim.tokens,
            logical_bytes: victim.logical_bytes,
            value,
        })
    }

    pub fn lookup_recurrent(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<E>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let matched_tokens = longest_component_prefix(root, tokens, ComponentKind::KvRecurrent)?;
        let entry = node_at_mut(root, &tokens[..matched_tokens])?
            .components
            .recurrent
            .as_mut()?;
        entry.last_used = self.clock;
        Some(RadixMatch {
            matched_tokens,
            stored_tokens: tokens[..matched_tokens].to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn acquire_recurrent(&mut self, namespace: &str, tokens: &[i32]) -> Option<RadixMatch<E>> {
        self.clock = self.clock.saturating_add(1);
        let root = self.roots.get_mut(namespace)?;
        let matched_tokens = longest_component_prefix(root, tokens, ComponentKind::KvRecurrent)?;
        let entry = node_at_mut(root, &tokens[..matched_tokens])?
            .components
            .recurrent
            .as_mut()?;
        entry.last_used = self.clock;
        entry.active_refs = entry.active_refs.saturating_add(1);
        Some(RadixMatch {
            matched_tokens,
            stored_tokens: tokens[..matched_tokens].to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    pub fn release_recurrent(&mut self, namespace: &str, tokens: &[i32]) -> bool {
        let Some(root) = self.roots.get_mut(namespace) else {
            return false;
        };
        let Some(entry) =
            node_at_mut(root, tokens).and_then(|node| node.components.recurrent.as_mut())
        else {
            return false;
        };
        if entry.active_refs == 0 {
            return false;
        }
        entry.active_refs -= 1;
        self.clock = self.clock.saturating_add(1);
        entry.last_used = self.clock;
        true
    }
}

#[derive(Debug)]
struct Victim {
    namespace: String,
    tokens: Vec<i32>,
    logical_bytes: u64,
    last_used: u64,
}

fn validate_tokens(tokens: &[i32]) -> Result<()> {
    if tokens.is_empty() {
        bail!("radix cache key must contain at least one token");
    }
    Ok(())
}

// Returning a value inserted through `BTreeMap::entry` keeps the entry borrow
// alive for `'a`, which prevents the split path from mutating the map again.
#[allow(clippy::map_entry)]
fn ensure_node<'a, R, E>(
    node: &'a mut RadixNode<R, E>,
    tokens: &[i32],
    splits: &mut u64,
) -> &'a mut RadixNode<R, E> {
    if tokens.is_empty() {
        return node;
    }
    let first = tokens[0];
    if !node.children.contains_key(&first) {
        node.children.insert(first, RadixNode::new(tokens.to_vec()));
        return node
            .children
            .get_mut(&first)
            .expect("new radix child should exist");
    }

    let common = {
        let child = node.children.get(&first).expect("radix child should exist");
        common_prefix_len(&child.edge, tokens)
    };
    let child_len = node
        .children
        .get(&first)
        .expect("radix child should exist")
        .edge
        .len();
    if common == child_len {
        let child = node
            .children
            .get_mut(&first)
            .expect("radix child should exist");
        return ensure_node(child, &tokens[common..], splits);
    }

    let mut existing = node
        .children
        .remove(&first)
        .expect("radix child should exist for split");
    let common_edge = existing.edge[..common].to_vec();
    existing.edge.drain(..common);
    let existing_first = existing.edge[0];

    let mut split = RadixNode::new(common_edge);
    *splits = splits.saturating_add(1);
    split.children.insert(existing_first, existing);
    if common == tokens.len() {
        node.children.insert(first, split);
        return node
            .children
            .get_mut(&first)
            .expect("split radix node should exist");
    }

    let new_edge = tokens[common..].to_vec();
    let new_first = new_edge[0];
    split.children.insert(new_first, RadixNode::new(new_edge));
    node.children.insert(first, split);
    node.children
        .get_mut(&first)
        .and_then(|split| split.children.get_mut(&new_first))
        .expect("new split radix child should exist")
}

/// Find the longest query prefix backed by any resident sequence.
///
/// Unlike recurrent checkpoints, native resident KV can be sliced when copied:
/// a cached sequence for `[1, 2, 3, 4]` can restore `[1, 2]` for a request that
/// diverges at token 3. Native restore only reads the stored resident source,
/// so multiple restores may hold references concurrently; eviction remains
/// blocked until every reference is released.
fn resident_backing_prefix<R, E>(
    root: &RadixNode<R, E>,
    tokens: &[i32],
) -> Option<(usize, Vec<i32>)> {
    let mut node = root;
    let mut remaining = tokens;
    let mut consumed = 0usize;
    let mut path = Vec::new();
    let mut best = None;

    if node.components.resident.is_some() {
        best = Some((0, Vec::new()));
    }

    while let Some(first) = remaining.first() {
        let Some(child) = node.children.get(first) else {
            // The query can diverge exactly at an existing branch point. No
            // child starts with its next token, but any resident descendant
            // still owns a native sequence that can be sliced to `consumed`.
            // Without this fallback, the first two branches split the edge
            // and every later sibling incorrectly becomes a full miss.
            if consumed > 0
                && let Some(stored_tokens) = nearest_resident_descendant(node, &path)
            {
                best = Some((consumed, stored_tokens));
            }
            break;
        };
        let common = common_prefix_len(&child.edge, remaining);
        if common < child.edge.len() {
            let mut child_path = path.clone();
            child_path.extend_from_slice(&child.edge);
            if common > 0
                && let Some(stored_tokens) = nearest_resident_descendant(child, &child_path)
            {
                best = Some((consumed.saturating_add(common), stored_tokens));
            }
            break;
        }
        consumed = consumed.saturating_add(common);
        remaining = &remaining[common..];
        path.extend_from_slice(&child.edge);
        node = child;
        if node.components.resident.is_some() {
            best = Some((consumed, path.clone()));
        }
    }

    if remaining.is_empty()
        && let Some(stored_tokens) = nearest_resident_descendant(node, &path)
    {
        best = Some((consumed, stored_tokens));
    }
    best.filter(|(matched, _)| *matched > 0)
}

fn nearest_resident_descendant<R, E>(
    node: &RadixNode<R, E>,
    node_path: &[i32],
) -> Option<Vec<i32>> {
    if node.components.resident.is_some() {
        return Some(node_path.to_vec());
    }
    for child in node.children.values() {
        let mut child_path = node_path.to_vec();
        child_path.extend_from_slice(&child.edge);
        if let Some(candidate) = nearest_resident_descendant(child, &child_path) {
            return Some(candidate);
        }
    }
    None
}

fn longest_component_prefix<R, E>(
    root: &RadixNode<R, E>,
    tokens: &[i32],
    component: ComponentKind,
) -> Option<usize> {
    let mut node = root;
    let mut remaining = tokens;
    let mut consumed = 0usize;
    let mut best = component_present(node, component).then_some(0);

    while let Some(first) = remaining.first() {
        let Some(child) = node.children.get(first) else {
            break;
        };
        let common = common_prefix_len(&child.edge, remaining);
        if common != child.edge.len() {
            break;
        }
        consumed = consumed.saturating_add(common);
        remaining = &remaining[common..];
        node = child;
        if component_present(node, component) {
            best = Some(consumed);
        }
    }
    best.filter(|matched| *matched > 0)
}

fn node_at_mut<'a, R, E>(
    mut node: &'a mut RadixNode<R, E>,
    mut tokens: &[i32],
) -> Option<&'a mut RadixNode<R, E>> {
    while !tokens.is_empty() {
        let child = node.children.get_mut(&tokens[0])?;
        if !tokens.starts_with(&child.edge) {
            return None;
        }
        tokens = &tokens[child.edge.len()..];
        node = child;
    }
    Some(node)
}

fn node_at<'a, R, E>(
    mut node: &'a RadixNode<R, E>,
    mut tokens: &[i32],
) -> Option<&'a RadixNode<R, E>> {
    while !tokens.is_empty() {
        let child = node.children.get(&tokens[0])?;
        if !tokens.starts_with(&child.edge) {
            return None;
        }
        tokens = &tokens[child.edge.len()..];
        node = child;
    }
    Some(node)
}

fn common_prefix_len(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn component_present<R, E>(node: &RadixNode<R, E>, component: ComponentKind) -> bool {
    match component {
        ComponentKind::ResidentKv => node.components.resident.is_some(),
        ComponentKind::KvRecurrent => node.components.recurrent.is_some(),
    }
}

fn find_resident_path<R, E>(
    node: &RadixNode<R, E>,
    path: &mut Vec<i32>,
    predicate: &mut impl FnMut(&R) -> bool,
) -> Option<(Vec<i32>, u64)> {
    let original_len = path.len();
    path.extend_from_slice(&node.edge);
    if let Some(entry) = node
        .components
        .resident
        .as_ref()
        .filter(|entry| predicate(&entry.value))
    {
        return Some((path.clone(), entry.logical_bytes));
    }
    for child in node.children.values() {
        if let Some(found) = find_resident_path(child, path, predicate) {
            return Some(found);
        }
    }
    path.truncate(original_len);
    None
}

fn find_recurrent_path<R, E>(
    node: &RadixNode<R, E>,
    path: &mut Vec<i32>,
    predicate: &mut impl FnMut(&E) -> bool,
) -> Option<(Vec<i32>, u64)> {
    let original_len = path.len();
    path.extend_from_slice(&node.edge);
    if let Some(entry) = node
        .components
        .recurrent
        .as_ref()
        .filter(|entry| predicate(&entry.value))
    {
        return Some((path.clone(), entry.logical_bytes));
    }
    for child in node.children.values() {
        if let Some(found) = find_recurrent_path(child, path, predicate) {
            return Some(found);
        }
    }
    path.truncate(original_len);
    None
}

fn collect_lru_victim<R, E>(
    namespace: &str,
    node: &RadixNode<R, E>,
    component: ComponentKind,
    path: &mut Vec<i32>,
    victim: &mut Option<Victim>,
) {
    let original_len = path.len();
    path.extend_from_slice(&node.edge);
    let candidate = match component {
        ComponentKind::ResidentKv => node
            .components
            .resident
            .as_ref()
            .filter(|entry| entry.active_refs == 0)
            .map(|entry| (entry.last_used, entry.logical_bytes)),
        ComponentKind::KvRecurrent => node
            .components
            .recurrent
            .as_ref()
            .filter(|entry| entry.active_refs == 0)
            .map(|entry| (entry.last_used, entry.logical_bytes)),
    };
    if let Some((last_used, logical_bytes)) = candidate {
        let replace = victim
            .as_ref()
            .map(|current| {
                (last_used, namespace, path.as_slice())
                    < (
                        current.last_used,
                        current.namespace.as_str(),
                        current.tokens.as_slice(),
                    )
            })
            .unwrap_or(true);
        if replace {
            *victim = Some(Victim {
                namespace: namespace.to_string(),
                tokens: path.clone(),
                logical_bytes,
                last_used,
            });
        }
    }
    for child in node.children.values() {
        collect_lru_victim(namespace, child, component, path, victim);
    }
    path.truncate(original_len);
}

fn collect_resident_eviction_candidates<R: Clone, E>(
    namespace: &str,
    node: &RadixNode<R, E>,
    path: &mut Vec<i32>,
    candidates: &mut Vec<RadixEvictionCandidate<R>>,
) {
    let original_len = path.len();
    path.extend_from_slice(&node.edge);
    if let Some(entry) = node
        .components
        .resident
        .as_ref()
        .filter(|entry| entry.active_refs == 0)
    {
        candidates.push(RadixEvictionCandidate {
            namespace: namespace.to_string(),
            tokens: path.clone(),
            logical_bytes: entry.logical_bytes,
            last_used: entry.last_used,
            value: entry.value.clone(),
        });
    }
    for child in node.children.values() {
        collect_resident_eviction_candidates(namespace, child, path, candidates);
    }
    path.truncate(original_len);
}

fn normalize_root<R, E>(root: &mut RadixNode<R, E>) {
    normalize_children(root);
}

fn normalize_children<R, E>(node: &mut RadixNode<R, E>) {
    let keys = node.children.keys().copied().collect::<Vec<_>>();
    for key in keys {
        let remove = {
            let child = node
                .children
                .get_mut(&key)
                .expect("radix child should exist during normalization");
            normalize_children(child);
            while child.components.resident.is_none()
                && child.components.recurrent.is_none()
                && child.children.len() == 1
            {
                let grandchild_key = *child
                    .children
                    .keys()
                    .next()
                    .expect("unary radix child should have one key");
                let mut grandchild = child
                    .children
                    .remove(&grandchild_key)
                    .expect("unary radix grandchild should exist");
                child.edge.append(&mut grandchild.edge);
                child.components = grandchild.components;
                child.children = grandchild.children;
            }
            child.is_empty()
        };
        if remove {
            node.children.remove(&key);
        }
    }
}

fn accumulate_stats<R, E>(
    node: &RadixNode<R, E>,
    is_root: bool,
    parent_depth: usize,
    stats: &mut UnifiedRadixCacheStats,
) {
    let depth = parent_depth.saturating_add(node.edge.len());
    if !is_root {
        stats.nodes = stats.nodes.saturating_add(1);
        stats.token_edges = stats.token_edges.saturating_add(node.edge.len());
    }
    if let Some(entry) = &node.components.resident {
        stats.resident_entries = stats.resident_entries.saturating_add(1);
        stats.resident_tokens = stats.resident_tokens.saturating_add(depth as u64);
        stats.resident_logical_bytes = stats
            .resident_logical_bytes
            .saturating_add(entry.logical_bytes);
        stats.resident_active_refs = stats
            .resident_active_refs
            .saturating_add(u64::from(entry.active_refs));
        if entry.active_refs > 0 {
            stats.resident_pinned_tokens =
                stats.resident_pinned_tokens.saturating_add(depth as u64);
        }
    }
    if let Some(entry) = &node.components.recurrent {
        stats.recurrent_entries = stats.recurrent_entries.saturating_add(1);
        stats.recurrent_logical_bytes = stats
            .recurrent_logical_bytes
            .saturating_add(entry.logical_bytes);
        stats.recurrent_active_refs = stats
            .recurrent_active_refs
            .saturating_add(u64::from(entry.active_refs));
    }
    for child in node.children.values() {
        accumulate_stats(child, false, depth, stats);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct DeterministicRng(u64);

    impl DeterministicRng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn below(&mut self, ceiling: usize) -> usize {
            (self.next() as usize) % ceiling
        }
    }

    fn random_key(rng: &mut DeterministicRng) -> (String, Vec<i32>) {
        let namespace = format!("model-{}", rng.below(3));
        let length = rng.below(8) + 1;
        let tokens = (0..length).map(|_| rng.below(6) as i32).collect();
        (namespace, tokens)
    }

    fn state_machine_budget() -> (usize, usize) {
        let seeds = std::env::var("SKIPPY_CACHE_STATE_MACHINE_SEEDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8)
            .clamp(1, 4_096);
        let steps = std::env::var("SKIPPY_CACHE_STATE_MACHINE_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2_000)
            .clamp(1, 100_000);
        (seeds, steps)
    }

    fn reference_longest(
        entries: &HashMap<(String, Vec<i32>), u64>,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<(usize, u64)> {
        entries
            .iter()
            .filter(|((candidate_namespace, candidate_tokens), _)| {
                candidate_namespace == namespace && tokens.starts_with(candidate_tokens)
            })
            .map(|((_, candidate_tokens), value)| (candidate_tokens.len(), *value))
            .max_by_key(|(matched, _)| *matched)
    }

    fn reference_resident_matched(
        entries: &HashMap<(String, Vec<i32>), u64>,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<usize> {
        entries
            .keys()
            .filter(|(candidate_namespace, _)| candidate_namespace == namespace)
            .map(|(_, candidate_tokens)| common_prefix_len(candidate_tokens, tokens))
            .filter(|matched| *matched > 0)
            .max()
    }

    #[test]
    fn longest_prefix_crosses_compressed_splits() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("model-a", &[1, 2, 3, 4], 40, "long")
            .unwrap();
        cache
            .insert_resident("model-a", &[1, 2], 20, "shared")
            .unwrap();
        cache
            .insert_resident("model-a", &[1, 9], 20, "branch")
            .unwrap();

        let exact = cache
            .lookup_resident("model-a", &[1, 2, 3, 4, 5])
            .expect("longest prefix hit");
        assert_eq!(exact.matched_tokens, 4);
        assert_eq!(exact.value, "long");

        let shared = cache
            .lookup_resident("model-a", &[1, 2, 8])
            .expect("shared prefix hit");
        assert_eq!(shared.matched_tokens, 2);
        assert_eq!(shared.value, "shared");
        assert_eq!(
            cache
                .lookup_resident("model-a", &[1, 7])
                .expect("branch-point common prefix")
                .matched_tokens,
            1
        );
        assert_eq!(cache.stats().splits, 2);
    }

    #[test]
    fn resident_kv_reuses_common_prefix_from_longer_cached_sequence() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("model-a", &[1, 2, 3, 4], 40, "long")
            .unwrap();

        let divergent = cache
            .lookup_resident("model-a", &[1, 2, 9, 10])
            .expect("common-prefix resident hit");
        assert_eq!(divergent.matched_tokens, 2);
        assert_eq!(divergent.stored_tokens, vec![1, 2, 3, 4]);
        assert_eq!(divergent.value, "long");

        let shorter = cache
            .lookup_resident("model-a", &[1, 2])
            .expect("shorter query resident hit");
        assert_eq!(shorter.matched_tokens, 2);
        assert_eq!(shorter.stored_tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn scheduler_peeks_do_not_heat_resident_lru() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache.insert_resident("stage", &[1, 2], 2, "old").unwrap();
        cache.insert_resident("stage", &[3, 4], 2, "new").unwrap();
        let epoch = cache.epoch();

        assert_eq!(
            cache.peek_resident("stage", &[1, 2, 9]).unwrap().value,
            "old"
        );
        assert_eq!(cache.epoch(), epoch);
        assert_eq!(cache.lru_resident_candidate().unwrap().value, "old");
    }

    #[test]
    fn scheduler_peeks_do_not_heat_recurrent_lru() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache.insert_recurrent("stage", &[1, 2], 2, "old").unwrap();
        cache.insert_recurrent("stage", &[3, 4], 2, "new").unwrap();
        let epoch = cache.epoch();

        assert_eq!(
            cache.peek_recurrent("stage", &[1, 2, 9]).unwrap().value,
            "old"
        );
        assert_eq!(cache.epoch(), epoch);
        assert_eq!(cache.lru_recurrent_candidate().unwrap().value, "old");
    }

    #[test]
    fn resident_kv_reuses_descendant_when_query_adds_a_new_branch() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("model-a", &[1, 2, 3, 4], 40, "left")
            .unwrap();
        cache
            .insert_resident("model-a", &[1, 2, 5, 6], 40, "right")
            .unwrap();

        let third_branch = cache
            .lookup_resident("model-a", &[1, 2, 7, 8])
            .expect("branch-point resident hit");
        assert_eq!(third_branch.matched_tokens, 2);
        assert!(
            third_branch.stored_tokens == vec![1, 2, 3, 4]
                || third_branch.stored_tokens == vec![1, 2, 5, 6]
        );
    }

    #[test]
    fn namespaces_are_hard_cache_boundaries() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("model-a", &[1, 2, 3], 3, "a")
            .unwrap();
        cache.insert_resident("model-b", &[1, 2], 2, "b").unwrap();

        assert_eq!(
            cache
                .lookup_resident("model-a", &[1, 2, 3, 4])
                .unwrap()
                .value,
            "a"
        );
        assert_eq!(
            cache
                .lookup_resident("model-b", &[1, 2, 3, 4])
                .unwrap()
                .value,
            "b"
        );
        assert!(cache.lookup_resident("model-c", &[1, 2, 3]).is_none());
    }

    #[test]
    fn resident_and_recurrent_components_share_one_logical_node() {
        let mut cache = UnifiedRadixCache::new();
        cache
            .insert_resident("stage", &[7, 8, 9], 10, "resident")
            .unwrap();
        cache
            .insert_recurrent("stage", &[7, 8, 9], 20, "recurrent")
            .unwrap();

        assert_eq!(
            cache
                .lookup_resident("stage", &[7, 8, 9, 10])
                .unwrap()
                .value,
            "resident"
        );
        assert_eq!(
            cache
                .lookup_recurrent("stage", &[7, 8, 9, 10])
                .unwrap()
                .value,
            "recurrent"
        );
        assert_eq!(
            cache.stats(),
            UnifiedRadixCacheStats {
                namespaces: 1,
                nodes: 1,
                token_edges: 3,
                resident_entries: 1,
                resident_tokens: 3,
                resident_logical_bytes: 10,
                recurrent_entries: 1,
                recurrent_logical_bytes: 20,
                ..UnifiedRadixCacheStats::default()
            }
        );
    }

    #[test]
    fn concurrent_resident_readers_share_source_and_protect_it_from_eviction() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache.insert_resident("stage", &[1], 10, "old").unwrap();
        cache.insert_resident("stage", &[2], 20, "new").unwrap();

        let first = cache
            .acquire_resident("stage", &[1, 9])
            .expect("first resident acquire");
        assert_eq!(first.active_refs, 1);
        let second = cache
            .acquire_resident("stage", &[1, 8])
            .expect("second resident acquire");
        assert_eq!(second.active_refs, 2);
        assert_eq!(
            cache
                .lookup_resident("stage", &[1, 7])
                .expect("active resident remains probe-visible")
                .active_refs,
            2
        );
        let evicted = cache.evict_lru_resident().expect("unreferenced victim");
        assert_eq!(evicted.tokens, vec![2]);
        assert_eq!(evicted.value, "new");
        assert_eq!(cache.stats().resident_evictions, 1);

        assert!(cache.release_resident("stage", &[1]));
        assert!(cache.evict_lru_resident().is_none());
        assert_eq!(cache.stats().resident_active_refs, 1);
        assert!(cache.release_resident("stage", &[1]));
        assert!(!cache.release_resident("stage", &[1]));
        assert_eq!(cache.evict_lru_resident().unwrap().value, "old");
        assert_eq!(cache.stats().resident_evictions, 2);
        assert_eq!(cache.stats().namespaces, 0);
    }

    #[test]
    fn capacity_candidates_exclude_pinned_entries_and_support_exact_eviction() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("stage", &[1, 2, 3], 30, "pinned")
            .unwrap();
        cache
            .insert_resident("stage", &[4, 5], 20, "evictable")
            .unwrap();
        cache.acquire_resident("stage", &[1, 2, 3]).unwrap();

        let candidates = cache.resident_eviction_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tokens, vec![4, 5]);
        assert_eq!(candidates[0].value, "evictable");
        assert_eq!(cache.stats().resident_pinned_tokens, 3);
        assert!(
            cache
                .evict_resident_candidate("stage", &[1, 2, 3])
                .is_none()
        );
        assert_eq!(
            cache
                .evict_resident_candidate("stage", &[4, 5])
                .unwrap()
                .value,
            "evictable"
        );
        assert_eq!(cache.stats().resident_evictions, 1);
    }

    #[test]
    fn occupied_resident_insert_cannot_replace_a_live_native_source() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache.insert_resident("stage", &[1], 10, "old").unwrap();
        cache.acquire_resident("stage", &[1]).unwrap();

        assert_eq!(
            cache
                .insert_resident("stage", &[1], 20, "replacement")
                .unwrap_err()
                .to_string(),
            "cannot replace an active resident radix entry"
        );
        assert_eq!(
            cache
                .insert_resident_if_vacant("stage", &[1], 20, "rejected")
                .unwrap(),
            Some("rejected")
        );
        let existing = cache.lookup_resident("stage", &[1]).unwrap();
        assert_eq!(existing.value, "old");
        assert_eq!(existing.active_refs, 1);
        assert!(cache.release_resident("stage", &[1]));
    }

    #[test]
    fn component_eviction_preserves_other_payload_at_same_prefix() {
        let mut cache = UnifiedRadixCache::new();
        cache
            .insert_resident("stage", &[1, 2], 10, "resident")
            .unwrap();
        cache
            .insert_recurrent("stage", &[1, 2], 20, "recurrent")
            .unwrap();

        assert_eq!(cache.evict_lru_resident().unwrap().value, "resident");
        assert!(cache.lookup_resident("stage", &[1, 2]).is_none());
        assert_eq!(
            cache.lookup_recurrent("stage", &[1, 2]).unwrap().value,
            "recurrent"
        );
        assert_eq!(cache.stats().nodes, 1);
    }

    #[test]
    fn removing_a_branch_prunes_and_recompresses_the_tree() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_resident("stage", &[1, 2, 3], 3, "left")
            .unwrap();
        cache
            .insert_resident("stage", &[1, 2, 4], 3, "right")
            .unwrap();
        assert_eq!(cache.stats().nodes, 3);

        assert_eq!(cache.remove_resident("stage", &[1, 2, 4]), Some("right"));
        assert_eq!(cache.stats().nodes, 1);
        assert_eq!(cache.stats().token_edges, 3);
        assert_eq!(
            cache.lookup_resident("stage", &[1, 2, 3, 5]).unwrap().value,
            "left"
        );
    }

    #[test]
    fn conditional_removal_requires_an_unreferenced_matching_entry() {
        let mut cache = UnifiedRadixCache::<&str, &str>::new();
        cache
            .insert_recurrent("stage", &[1, 2], 2, "payload-a")
            .unwrap();
        cache.acquire_recurrent("stage", &[1, 2]).unwrap();

        assert_eq!(
            cache.remove_recurrent_if("stage", &[1, 2], |value| *value == "payload-a"),
            None,
            "an active restore lease must prevent quarantine"
        );
        assert!(cache.release_recurrent("stage", &[1, 2]));
        assert_eq!(
            cache
                .insert_recurrent("stage", &[1, 2], 2, "payload-b")
                .unwrap(),
            Some("payload-a")
        );
        assert_eq!(
            cache.remove_recurrent_if("stage", &[1, 2], |value| *value == "payload-a"),
            None,
            "an old restore failure must not remove a replacement"
        );
        assert_eq!(
            cache.remove_recurrent_if("stage", &[1, 2], |value| *value == "payload-b"),
            Some("payload-b")
        );
    }

    #[test]
    fn recurrent_insert_cannot_replace_an_active_restore_source() {
        let mut cache = UnifiedRadixCache::<(), &str>::new();
        cache
            .insert_recurrent("stage", &[1, 2], 2, "original")
            .unwrap();
        cache.acquire_recurrent("stage", &[1, 2]).unwrap();

        assert_eq!(
            cache
                .insert_recurrent("stage", &[1, 2], 2, "replacement")
                .unwrap_err()
                .to_string(),
            "cannot replace an active recurrent radix entry"
        );
        assert_eq!(
            cache.lookup_recurrent("stage", &[1, 2]).unwrap().value,
            "original"
        );
        assert!(cache.release_recurrent("stage", &[1, 2]));
    }

    #[test]
    fn empty_prefixes_are_rejected() {
        let mut cache = UnifiedRadixCache::<(), ()>::new();
        assert_eq!(
            cache
                .insert_resident("stage", &[], 0, ())
                .unwrap_err()
                .to_string(),
            "radix cache key must contain at least one token"
        );
        assert_eq!(
            cache
                .insert_recurrent("stage", &[], 0, ())
                .unwrap_err()
                .to_string(),
            "radix cache key must contain at least one token"
        );
    }

    #[test]
    fn randomized_operations_match_flat_longest_prefix_reference() {
        let mut rng = DeterministicRng(0x5a17_cafe_d00d_f00d);
        let mut cache = UnifiedRadixCache::<u64, u64>::new();
        let mut resident = HashMap::<(String, Vec<i32>), u64>::new();
        let mut recurrent = HashMap::<(String, Vec<i32>), u64>::new();

        for step in 0..10_000_u64 {
            let (namespace, tokens) = random_key(&mut rng);
            match rng.below(8) {
                0 | 1 => {
                    let value = step ^ 0x55aa;
                    let expected = resident.insert((namespace.clone(), tokens.clone()), value);
                    assert_eq!(
                        cache
                            .insert_resident(namespace, &tokens, tokens.len() as u64, value)
                            .unwrap(),
                        expected
                    );
                }
                2 | 3 => {
                    let value = step ^ 0xaa55;
                    let expected = recurrent.insert((namespace.clone(), tokens.clone()), value);
                    assert_eq!(
                        cache
                            .insert_recurrent(namespace, &tokens, tokens.len() as u64, value)
                            .unwrap(),
                        expected
                    );
                }
                4 => {
                    let expected = resident.remove(&(namespace.clone(), tokens.clone()));
                    assert_eq!(cache.remove_resident(&namespace, &tokens), expected);
                }
                5 => {
                    let expected = recurrent.remove(&(namespace.clone(), tokens.clone()));
                    assert_eq!(cache.remove_recurrent(&namespace, &tokens), expected);
                }
                6 => {
                    let expected = reference_resident_matched(&resident, &namespace, &tokens);
                    let actual = cache
                        .lookup_resident(&namespace, &tokens)
                        .map(|hit| hit.matched_tokens);
                    assert_eq!(actual, expected);
                }
                7 => {
                    let expected = reference_longest(&recurrent, &namespace, &tokens);
                    let actual = cache
                        .lookup_recurrent(&namespace, &tokens)
                        .map(|hit| (hit.matched_tokens, hit.value));
                    assert_eq!(actual, expected);
                }
                _ => unreachable!(),
            }

            if step % 97 == 0 {
                let stats = cache.stats();
                assert_eq!(stats.resident_entries, resident.len());
                assert_eq!(stats.recurrent_entries, recurrent.len());
            }
        }
    }

    #[test]
    fn randomized_leases_and_evictions_preserve_ownership_stats() {
        #[derive(Clone, Copy, Default)]
        struct Entry {
            value: u64,
            bytes: u64,
            refs: u32,
        }

        let (seed_count, steps) = state_machine_budget();
        for seed_index in 0..seed_count {
            let seed = 0x19a4_7c2e_d18b_5601_u64
                .wrapping_add((seed_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut rng = DeterministicRng(seed);
            let mut cache = UnifiedRadixCache::<u64, u64>::new();
            let mut resident = HashMap::<i32, Entry>::new();
            let mut recurrent = HashMap::<i32, Entry>::new();
            let mut resident_evictions = 0_u64;
            let mut recurrent_evictions = 0_u64;

            for step in 0..steps {
                let token = (rng.below(32) + 1) as i32;
                let tokens = [token];
                let value = (seed_index as u64) << 32 | step as u64;
                let bytes = (rng.below(8) + 1) as u64;
                match rng.below(8) {
                    0 => {
                        let active = resident.get(&token).is_some_and(|entry| entry.refs > 0);
                        let result = cache.insert_resident("state", &tokens, bytes, value);
                        if active {
                            assert!(result.is_err(), "seed={seed:#x} step={step}");
                        } else {
                            let expected = resident.insert(
                                token,
                                Entry {
                                    value,
                                    bytes,
                                    refs: 0,
                                },
                            );
                            assert_eq!(
                                result.unwrap(),
                                expected.map(|entry| entry.value),
                                "seed={seed:#x} step={step}"
                            );
                        }
                    }
                    1 => {
                        let active = recurrent.get(&token).is_some_and(|entry| entry.refs > 0);
                        let result = cache.insert_recurrent("state", &tokens, bytes, value);
                        if active {
                            assert!(result.is_err(), "seed={seed:#x} step={step}");
                        } else {
                            let expected = recurrent.insert(
                                token,
                                Entry {
                                    value,
                                    bytes,
                                    refs: 0,
                                },
                            );
                            assert_eq!(
                                result.unwrap(),
                                expected.map(|entry| entry.value),
                                "seed={seed:#x} step={step}"
                            );
                        }
                    }
                    2 => {
                        let actual = cache.acquire_resident("state", &tokens);
                        let expected = resident.get_mut(&token);
                        assert_eq!(
                            actual.is_some(),
                            expected.is_some(),
                            "seed={seed:#x} step={step}"
                        );
                        if let (Some(hit), Some(entry)) = (actual, expected) {
                            entry.refs = entry.refs.saturating_add(1);
                            assert_eq!(hit.value, entry.value, "seed={seed:#x} step={step}");
                            assert_eq!(hit.active_refs, entry.refs, "seed={seed:#x} step={step}");
                        }
                    }
                    3 => {
                        let actual = cache.acquire_recurrent("state", &tokens);
                        let expected = recurrent.get_mut(&token);
                        assert_eq!(
                            actual.is_some(),
                            expected.is_some(),
                            "seed={seed:#x} step={step}"
                        );
                        if let (Some(hit), Some(entry)) = (actual, expected) {
                            entry.refs = entry.refs.saturating_add(1);
                            assert_eq!(hit.value, entry.value, "seed={seed:#x} step={step}");
                            assert_eq!(hit.active_refs, entry.refs, "seed={seed:#x} step={step}");
                        }
                    }
                    4 => {
                        let expected = resident.get_mut(&token).is_some_and(|entry| {
                            if entry.refs == 0 {
                                false
                            } else {
                                entry.refs -= 1;
                                true
                            }
                        });
                        assert_eq!(
                            cache.release_resident("state", &tokens),
                            expected,
                            "seed={seed:#x} step={step}"
                        );
                    }
                    5 => {
                        let expected = recurrent.get_mut(&token).is_some_and(|entry| {
                            if entry.refs == 0 {
                                false
                            } else {
                                entry.refs -= 1;
                                true
                            }
                        });
                        assert_eq!(
                            cache.release_recurrent("state", &tokens),
                            expected,
                            "seed={seed:#x} step={step}"
                        );
                    }
                    6 => {
                        let expected = resident
                            .get(&token)
                            .filter(|entry| entry.refs == 0)
                            .copied();
                        let actual = cache.evict_resident_candidate("state", &tokens);
                        assert_eq!(
                            actual.is_some(),
                            expected.is_some(),
                            "seed={seed:#x} step={step}"
                        );
                        if let (Some(eviction), Some(entry)) = (actual, expected) {
                            assert_eq!(eviction.value, entry.value, "seed={seed:#x} step={step}");
                            resident.remove(&token);
                            resident_evictions += 1;
                        }
                    }
                    7 => {
                        let actual = cache.evict_lru_recurrent();
                        let has_candidate = recurrent.values().any(|entry| entry.refs == 0);
                        assert_eq!(
                            actual.is_some(),
                            has_candidate,
                            "seed={seed:#x} step={step}"
                        );
                        if let Some(eviction) = actual {
                            let evicted_token = eviction.tokens[0];
                            let entry = recurrent
                                .remove(&evicted_token)
                                .expect("evicted recurrent entry must exist in reference model");
                            assert_eq!(entry.refs, 0, "seed={seed:#x} step={step}");
                            assert_eq!(eviction.value, entry.value, "seed={seed:#x} step={step}");
                            recurrent_evictions += 1;
                        }
                    }
                    _ => unreachable!(),
                }

                let stats = cache.stats();
                assert_eq!(
                    stats.namespaces,
                    usize::from(!resident.is_empty() || !recurrent.is_empty()),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_entries,
                    resident.len(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_tokens,
                    resident.len() as u64,
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.recurrent_entries,
                    recurrent.len(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_pinned_tokens,
                    resident.values().filter(|entry| entry.refs > 0).count() as u64,
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_logical_bytes,
                    resident.values().map(|entry| entry.bytes).sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.recurrent_logical_bytes,
                    recurrent.values().map(|entry| entry.bytes).sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_active_refs,
                    resident
                        .values()
                        .map(|entry| u64::from(entry.refs))
                        .sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.recurrent_active_refs,
                    recurrent
                        .values()
                        .map(|entry| u64::from(entry.refs))
                        .sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.resident_evictions, resident_evictions,
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    stats.recurrent_evictions, recurrent_evictions,
                    "seed={seed:#x} step={step}"
                );
            }
        }
    }
}
