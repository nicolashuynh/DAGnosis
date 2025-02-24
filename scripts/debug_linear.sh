python ../experiments/synthetic/generate_data.py sem_type=linear name_experiment=debug n_repetitions=1
python ../experiments/synthetic/train_cp.py name_experiment=debug n_repetitions=1
python ../experiments/synthetic/test_cp.py name_experiment=debug n_repetitions=1