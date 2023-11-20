. dataset.config

python3 make_datasets.py $raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $n_utt_attr

python3 sample_dataset.py $data_dir/train.pkl $data_dir/train_speaker2filenames.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size

python3 sample_dataset.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $data_dir/train_speaker2filenames.pkl $testing_samples $segment_size
python3 sample_dataset.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $data_dir/train_speaker2filenames.pkl $testing_samples $segment_size
