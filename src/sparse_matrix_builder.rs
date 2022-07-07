use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use log::info;
use rustc_hash::FxHasher;

use crate::configuration::Column;
use crate::sparse_matrix::SparseMatrixReader;
use crate::sparse_matrix::{Entry, Hash, SparseMatrix};

/// Creates combinations of column pairs as sparse matrices.
/// Let's say that we have such columns configuration: complex::a reflexive::complex::b c. This is provided
/// as `&[Column]` after parsing the config.
/// The allowed column modifiers are:
/// - transient - the field is virtual - it is considered during embedding process, no entity is written for the column,
/// - complex   - the field is composite, containing multiple entity identifiers separated by space,
/// - reflexive - the field is reflexive, which means that it interacts with itself, additional output file is written for every such field.
/// We create sparse matrix for every columns relations (based on column modifiers).
/// For our example we have:
/// - sparse matrix for column a and b,
/// - sparse matrix for column a and c,
/// - sparse matrix for column b and c,
/// - sparse matrix for column b and b (reflexive column).
/// Apart from column names in sparse matrix we provide indices for incoming data. We have 3 columns such as a, b and c
/// but column b is reflexive so we need to include this column. The result is: (a, b, c, b).
/// The rule is that every reflexive column is append with the order of occurrence to the end of constructed array.
pub fn create_sparse_matrices_builders(cols: &[Column]) -> Vec<SparseMatrixBuilder> {
    let mut sparse_matrix_builders: Vec<SparseMatrixBuilder> = Vec::new();
    let num_fields = cols.len();
    let mut reflexive_count = 0;

    for i in 0..num_fields {
        for j in i..num_fields {
            let col_i = &cols[i];
            let col_j = &cols[j];
            if i < j && !(col_i.transient && col_j.transient) {
                let sm = SparseMatrixBuilder::new(
                    i as u8,
                    col_i.name.clone(),
                    j as u8,
                    col_j.name.clone(),
                );
                sparse_matrix_builders.push(sm);
            } else if i == j && col_i.reflexive {
                let new_j = num_fields + reflexive_count;
                reflexive_count += 1;
                let sm = SparseMatrixBuilder::new(
                    i as u8,
                    col_i.name.clone(),
                    new_j as u8,
                    col_j.name.clone(),
                );
                sparse_matrix_builders.push(sm);
            }
        }
    }
    sparse_matrix_builders
}

#[derive(Default, Debug, Clone, Copy)]
struct Entity {
    pub occurrence: u32,
    pub row_sum: f32,
    pub index: u32, // set in second stage
}

impl Entity {
    pub fn new(initial_value: f32) -> Self {
        Self {
            occurrence: 1,
            row_sum: initial_value,
            index: 0,
        }
    }
}

#[derive(Debug)]
struct Edge {
    value: f32,
}

#[derive(Debug)]
pub struct SparseMatrixBuilder {
    /// First column index for which we creates subgraph
    pub col_a_id: u8,

    /// First column name
    pub col_a_name: String,

    /// Second column index for which we creates subgraph
    pub col_b_id: u8,

    /// Second column name
    pub col_b_name: String,

    edge_count: AtomicU32,
    hash_2_row: DashMap<u64, Entity, BuildHasherDefault<FxHasher>>,
    hashes_2_edge: DashMap<(u64, u64), Edge, BuildHasherDefault<FxHasher>>,
}

impl SparseMatrixBuilder {
    pub fn new(col_a_id: u8, col_a_name: String, col_b_id: u8, col_b_name: String) -> Self {
        Self {
            col_a_id,
            col_a_name,
            col_b_id,
            col_b_name,
            hash_2_row: DashMap::default(),
            hashes_2_edge: DashMap::default(),
            edge_count: AtomicU32::default(),
        }
    }

    pub fn handle_pair(&self, hashes: &[u64]) {
        let a = self.col_a_id;
        let b = self.col_b_id;
        self.add_pair_symmetric(
            hashes[(a + 1) as usize],
            hashes[(b + 1) as usize],
            hashes[0],
        );
    }

    fn add_pair_symmetric(&self, a_hash: u64, b_hash: u64, count: u64) {
        let value = 1f32 / (count as f32);

        self.update_row(a_hash, value);
        self.update_row(b_hash, value);

        self.edge_count.fetch_add(1, Ordering::Relaxed);

        self.update_edge(a_hash, b_hash, value);
        self.update_edge(b_hash, a_hash, value);
    }

    fn update_row(&self, hash: u64, val: f32) {
        match self.hash_2_row.entry(hash) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(Entity::new(val));
            }
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                let row = entry.get_mut();
                row.occurrence += 1;
                row.row_sum += val;
            }
        }
    }

    fn update_edge(&self, a_hash: u64, b_hash: u64, val: f32) {
        match self.hashes_2_edge.entry((a_hash, b_hash)) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(Edge { value: val });
            }
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                entry.get_mut().value += val;
            }
        };
    }

    pub fn finish(mut self) -> SparseMatrix {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;

        self.hash_2_row
            .iter_mut()
            .enumerate()
            .for_each(|(ix, mut e)| {
                let mut row = e.value_mut();
                row.index = ix as u32
            });

        let entities = self
            .hashes_2_edge
            .par_iter()
            .map(|e| {
                let (row_hash, col_hash) = e.key();
                let row_entity = self.hash_2_row.get(row_hash).unwrap();
                let col_entity = self.hash_2_row.get(col_hash).unwrap();

                let normalized_edge_value = e.value / row_entity.row_sum;

                Entry {
                    row: row_entity.index,
                    col: col_entity.index,
                    value: normalized_edge_value,
                }
            })
            .collect();

        let hashes = self
            .hash_2_row
            .par_iter()
            .map(|e| {
                let entity_hash = e.key();
                let entity = e.value();

                Hash {
                    value: *entity_hash,
                    id: entity.index,
                    occurrence: entity.occurrence,
                }
            })
            .collect();

        let sparse_matrix = SparseMatrix::new(
            self.col_a_id,
            self.col_a_name,
            self.col_b_id,
            self.col_b_name,
            hashes,
            entities,
        );

        info!(
            "Number of entities: {}",
            sparse_matrix.get_number_of_entities()
        );
        info!("Number of edges: {}", self.edge_count.get_mut());
        info!(
            "Number of entries: {}",
            sparse_matrix.get_number_of_entries()
        );

        // let hash_2_id_mem_size = self.hash_2_id.capacity() * 12;
        // let hash_mem_size = mem::size_of::<Hash>();
        // let id_2_hash_mem_size = self.id_2_hash.read().unwrap().capacity() * hash_mem_size;
        // let row_sum_mem_size = self.row_sum.read().unwrap().capacity() * 4;
        // let pair_index_mem_size = self.pair_index.capacity() * 12;
        //
        // let entry_mem_size = mem::size_of::<Entry>();
        // let entries_mem_size = self.entries.read().unwrap().capacity() * entry_mem_size;

        // let total_mem_size = hash_2_id_mem_size
        //     + id_2_hash_mem_size
        //     + row_sum_mem_size
        //     + pair_index_mem_size
        //     + entries_mem_size;
        //
        // info!(
        //     "Total memory usage by the struct ~ {} MB",
        //     (total_mem_size / 1048576)
        // );

        sparse_matrix
    }
}
