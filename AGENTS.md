The scripts starting with llama3_ are the tools to calculate/simulate the communication time for different collective algorithms and reconfigration plans during LLM training. With the script getting bigger, you may break one script into several different modules, but existing functionality should be kept immutable if also used by older scripts to keep behavior consistent. By creating different entry scripts calling different functions, we can switch different simulation behaviors.

If you create functions for new scripts inside llama3_comm/, please state in the comments which script introduced it.

If you are exploring the codebase, please also identify the correct module and function actually used in the given entry script, so that you can understand the codebase better.

The scripts starting with llama3_ are sometimes legacy compared to newer scripts. Please read README.md to find out the functionality of the scripts, and base your analysis on the most advanced / latest scripts.