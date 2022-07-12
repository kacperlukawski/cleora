use std::cmp::min;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::configuration::{Column, Configuration, FileType, OutputFormat};
use crate::embedding::{calculate_embeddings, calculate_embeddings_mmap};
use crate::entity::{EntityProcessor, SMALL_VECTOR_SIZE};
use crate::persistence::embedding::{EmbeddingPersistor, NpyPersistor, TextFileVectorPersistor};
use crate::persistence::entity::InMemoryEntityMappingPersistor;
use crate::sparse_matrix::SparseMatrix;
use crate::sparse_matrix_builder::create_sparse_matrices_builders;
use crossbeam::channel;
use crossbeam::thread as cb_thread;
use log::{error, info, warn};
use num_cpus;
use simdjson_rust::dom;
use smallvec::{smallvec, SmallVec};
use std::sync::Arc;
use std::thread;

/// Create SparseMatrix'es based on columns config. Every SparseMatrix operates in separate
/// thread. EntityProcessor reads data in main thread and broadcast cartesian products
/// to SparseMatrix'es.
pub fn build_graphs(
    config: &Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
) -> Vec<SparseMatrix> {
    let sparse_matrices = create_sparse_matrices_builders(&config.columns);
    dbg!(&sparse_matrices);

    let sparse_matrices_ref = &sparse_matrices;

    cb_thread::scope(|s| {
        let processing_worker_num = num_cpus::get();

        let (hyperedges_s, hyperedges_r) = channel::bounded(processing_worker_num * 16);

        for _ in 0..processing_worker_num {
            let receiver = hyperedges_r.clone();
            let entity_processor = EntityProcessor::new(
                config,
                in_memory_entity_mapping_persistor.clone(),
            );

            s.spawn(move |_| {
                for row in receiver {
                    let row: Vec<SmallVec<[_; SMALL_VECTOR_SIZE]>> = row;
                    for hashes in entity_processor.process_row_and_get_edges(&row) {
                        for sparse_matrix in sparse_matrices_ref {
                            sparse_matrix.handle_pair(&hashes)
                        }
                    }
                }
            });
        }

        {
            let (files_s, files_r) = channel::unbounded();
            for input in &config.input {
                files_s.send(input).unwrap()
            }

            let file_reading_worker_num = min(4, config.input.len());

            for _ in 0..file_reading_worker_num {
                let row_s = hyperedges_s.clone();
                let files_r = files_r.clone();

                match &config.file_type {
                    FileType::Json => {
                        s.spawn(move |_| {
                            let mut parser = dom::Parser::default();

                            for input in files_r {
                                read_file(input, config.log_every_n as u64, |line| {
                                    let row = parse_json_line(line, &mut parser, &config.columns);
                                    row_s.send(row).unwrap();
                                })
                            }
                        });
                    }
                    FileType::Tsv => {
                        let config_col_num = config.columns.len();
                        s.spawn(move |_| {
                            for input in files_r {
                                read_file(input, config.log_every_n as u64, |line| {
                                    let row = parse_tsv_line(line);
                                    let line_col_num = row.len();
                                    if line_col_num == config_col_num {
                                        row_s.send(row).unwrap();
                                    } else {
                                        warn!("Wrong number of columns (expected: {}, provided: {}). The line [{}] is skipped.", config_col_num, line_col_num, line);
                                    }
                                });
                            }
                        });
                    }
                }
            }
        }

        // Drop it so all channels are closed when work is finished
        drop(hyperedges_s);
        drop(hyperedges_r);
    }).expect("Threads finished work");

    sparse_matrices
        .into_iter()
        .map(|smb| smb.finish())
        .collect()
}

/// Read file line by line. Pass every valid line to handler for parsing.
fn read_file<F>(filepath: &str, log_every: u64, mut line_handler: F)
where
    F: FnMut(&str),
{
    let input_file = File::open(filepath).expect("Can't open file");
    let mut buffered = BufReader::new(input_file);

    let mut line_number = 1u64;
    let mut line = String::new();
    loop {
        match buffered.read_line(&mut line) {
            Ok(bytes_read) => {
                // EOF
                if bytes_read == 0 {
                    break;
                }

                line_handler(&line);
            }
            Err(err) => {
                error!("Can't read line number: {}. Error: {}.", line_number, err);
            }
        };

        // clear to reuse the buffer
        line.clear();

        if line_number % log_every == 0 {
            info!("Number of lines processed: {}", line_number);
        }

        line_number += 1;
    }
}

/// Parse a line of JSON and read its columns into a vector for processing.
fn parse_json_line(
    line: &str,
    parser: &mut dom::Parser,
    columns: &[Column],
) -> Vec<SmallVec<[String; SMALL_VECTOR_SIZE]>> {
    let parsed = parser.parse(line).unwrap();
    columns
        .iter()
        .map(|c| {
            if !c.complex {
                let elem = parsed.at_key(&c.name).unwrap();
                let value = match elem.get_type() {
                    dom::element::ElementType::String => elem.get_string().unwrap(),
                    _ => elem.minify(),
                };
                smallvec![value]
            } else {
                parsed
                    .at_key(&c.name)
                    .unwrap()
                    .get_array()
                    .expect("Values for complex columns must be arrays")
                    .into_iter()
                    .map(|v| match v.get_type() {
                        dom::element::ElementType::String => v.get_string().unwrap(),
                        _ => v.minify(),
                    })
                    .collect()
            }
        })
        .collect()
}

/// Parse a line of TSV and read its columns into a vector for processing.
fn parse_tsv_line(line: &str) -> Vec<SmallVec<[String; SMALL_VECTOR_SIZE]>> {
    let values = line.trim().split('\t');
    values
        .map(|c| c.split(' ').map(|s| s.to_owned()).collect())
        .collect()
}

/// Train SparseMatrix'es (graphs) in separated threads.
pub fn train(
    config: Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
    sparse_matrices: Vec<SparseMatrix>,
) {
    let config = Arc::new(config);
    let mut embedding_threads = Vec::new();
    for sparse_matrix in sparse_matrices {
        let sparse_matrix = Arc::new(sparse_matrix);
        let config = config.clone();
        let in_memory_entity_mapping_persistor = in_memory_entity_mapping_persistor.clone();
        let handle = thread::spawn(move || {
            let directory = match config.output_dir.as_ref() {
                Some(out) => format!("{}/", out.clone()),
                None => String::from(""),
            };
            let ofp = format!(
                "{}{}__{}__{}.out",
                directory,
                config.relation_name,
                sparse_matrix.col_a_name.as_str(),
                sparse_matrix.col_b_name.as_str()
            );

            let mut persistor: Box<dyn EmbeddingPersistor> = match &config.output_format {
                OutputFormat::TextFile => Box::new(TextFileVectorPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
                OutputFormat::Numpy => Box::new(NpyPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
            };
            if config.in_memory_embedding_calculation {
                calculate_embeddings(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            } else {
                calculate_embeddings_mmap(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            }
        });
        embedding_threads.push(handle);
    }

    for join_handle in embedding_threads {
        join_handle
            .join()
            .expect("Couldn't join on the associated thread");
    }
}
