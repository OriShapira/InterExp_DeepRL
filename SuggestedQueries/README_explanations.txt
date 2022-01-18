- test_full.py: Tests a trained suggested queries model on DUC 2006 with our metrics (against ref summs), and DUC 2001 with the standard F1 metrics (against gold KPs)
	- Puts the output suggested queries and results in the test_results folder under the model directory
	- Any split can be run on (including train and val in case needed). If 'test', it also means to run on 'test_kp' (2001 KP test set)

- test_full_all_saved_models.py: Tests (using above script) all models in the saved_models_path folder
	- if --recompute passed in, recomputes, even if results are already found in models' folders
	- set which splits to compute for the models in splits_to_compute


- dataset/prepare_mds_kps.py: Prepares the mutli-document KPs for the DUC 2001 data (based on the single document KPs)
	- This can be called from outside with the prepare_data_in_datadir function, which creates a folder in base_test_data_dir, in the expected topics+samples structure for reading from the model code
