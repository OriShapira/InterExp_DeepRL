# Interactive Expansion-Based Summarization with Deep Reinforcement Learning

A deep RL based solution for the [interactive expansion-based summarization task](https://aclanthology.org/2021.naacl-main.54.pdf).
Two subtasks are addressed: query-assisted summarization and suggested-queries extraction. Each subtask has an implemented model here.

Resources for the paper: [Interactive Query-Assisted Summarization via Deep Reinforcement Learning](https://todo)

Our deep RL models are based on the [RL-MMR system](https://aclanthology.org/2020.emnlp-main.136.pdf), borrowing code from their [repository](https://github.com/morningmoni/RL-MMR).

We use the evaluation framework from [here](https://github.com/OriShapira/InterExp) to collect user sessions via controlled crowdsourcing and evaluate the sessions.

## General Info
Purpose of the task:
- Explore a document set interactively.
- An InterExp system provides an initial summary, and then expands on it in response to queries submitted by the user.
- See [demo](https://nlp.biu.ac.il/~oris/qfse/qfse.html).

Query-assisted summarization:
- Input:
	- document set on a topic
	- textual query (can be empty for a generic summary)
	- history of texts seen (can be empty)
	- length of summary to output
- Ouput: query-biased summary on the topic, accounting for the history, with the requested length
	- if the query and history are empty, the summary is a standard generic summary
	
Suggested-queries extraction:
- Input:
	- document set on a topic
	- history of texts seen (can be empty)
	- number of suggestions to output
- Output: list of suggested queries (short phrases) for the topic, accounting for the history, with the requested number of suggestions



## Code and Folders in the Project

### Summarization
Query-assisted summarization model code.
- To train: ```python train_full_rl.py <args>```
	- Example: ```python train_full_rl.py --reward_summ r1r --reward_query_every 2 --query_encoding_in_input true --ignore_queries_summ_reward true --beta 0.5```
- To test: there are several ways to test a trained model (see individual scripts for required arguments)
	- Command line interaction application: test_interaction.py
	- Full dataset test (validation or test sets): test_full.py
	- Test on sessions collected previously (like simulations): test_session_data.py
	
### SuggestedQueries
Suggested queries extraction model code.
- To train: ```python train_full_rl.py <args>```
	- Example: ```python train_full_rl.py --phrasing_method posregex --max_num_steps 20 --reward_func tf --weight_initial 0.5 --weight_preceding 0.5 --stem True --remove_stop True --beta 0.5```
- To test a trained model: test_full.py

### QFSE
Based on the application from the [InterExp respository](https://github.com/OriShapira/InterExp), we implemented the summarizer and suggested query extractor modules for our models. Some minor code modifications were made to original files to accomodate changes for our models.

### WebApp
Also based on the application from the [InterExp respository](https://github.com/OriShapira/InterExp), we updated the application to support updating suggested queries, since the original method only used a static list of suggested queries initialized once.

### Crowdsourcing
See the [InterExp respository](https://github.com/OriShapira/InterExp) for the controlled crowdsourcing resources.
Here there are collected sessions:
- BaselineOriginal: the sessions from the InterExp project, used for simulation testing at Summarization/test_session_data.py
- Collected: the sessions we collected in this project, comparing two variants of our system to a baseline from the InterExp project.
	- experiment1 was run with summarization saved_model Apr21_12-00-53 (only uses the input query at inference time in the qMMR function)
	- experiment1 was run with summarization saved_model Apr12_23-23-47 (injects the query at train time in all components, and also at inference time)
	- both experiments used suggested queries saved_model kp_Jul07_17-56-44 (reward tf, gamma1=0.5, gamma2=0.9, stemming and stop word removal)
	
## Citation
If any resource from this repository is used, please cite:
```
TODO
```