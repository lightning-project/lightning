use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
struct Entry<K, V>(K, V);

impl<K: Ord, V> Ord for Entry<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&other.0, &self.0) // note: reversed arguments
    }
}

impl<K: Ord, V> PartialOrd for Entry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl<K: Eq, V> Eq for Entry<K, V> {}
impl<K: PartialEq, V> PartialEq for Entry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.0, &other.0)
    }
}

#[derive(Debug, Clone)]
pub struct OrderedQueue<K, V>(BinaryHeap<Entry<K, V>>);

impl<K, V> OrderedQueue<K, V>
where
    K: Ord,
{
    pub fn new() -> Self {
        Self(BinaryHeap::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(BinaryHeap::with_capacity(capacity))
    }

    pub fn push(&mut self, key: K, value: V) {
        self.0.push(Entry(key, value));
    }

    pub fn pop_min(&mut self) -> Option<(K, V)> {
        match self.0.pop() {
            Some(Entry(k, v)) => Some((k, v)),
            None => None,
        }
    }

    pub fn peek_min(&mut self) -> Option<(&K, &V)> {
        match self.0.peek() {
            Some(Entry(k, v)) => Some((k, v)),
            None => None,
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<K: Ord, V> Default for OrderedQueue<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
