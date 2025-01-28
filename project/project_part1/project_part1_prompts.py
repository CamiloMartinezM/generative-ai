system_message_nus = "You are an AI tutor. You have to help a student learning programming. The program uses Python. You have to strictly follow the format for the final output as instructed below."

user_message_nus_repair_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]


Fix the buggy code. Output your entire fixed code between [FIXED] and [/FIXED].
"""

user_message_nus_hint_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]

Provide a concise single-sentence hint to the student about one bug in the student's buggy code. Output your hint between [HINT] and [/HINT].
"""

user_message_nus_hint_advanced = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]

Previously, a Large Language Model was asked to review the student's buggy code and generate a repaired program based on the problem description. The repaired program is as follows:

[Repaired Code Starts]
{repaired_program}
[Repaired Code Ends]

Nevertheless, this repaired program is not guaranteed to be correct. For you to review whether this repaired program is correct or not, the results of the test cases for this problem are provided below:

[Testcases results of repaired code Starts]
{testcases_results}
[Testcases results of repaired code Ends]

Based on all of the above information, create an explanation written inside placeholders [EXP] and [/EXP] that analyzes the student's buggy code, the provided repaired program and the testcases results, to come up step-by-step with the best approach to provide a hint to the student. After that, provide a concise single-sentence hint to the student about one bug in the student's buggy code, which allows him to get closer to the correct program, but without directly giving him the code or detailed explanations. Output your hint between [HINT] and [/HINT]. Make sure to always use the [HINT] and [/HINT] placeholders with a hint inside. Do not output anything else inside the [HINT] placeholders that is not part of your chosen hint. Do not refer to the student as "the student", but with "you" if strictly necessary. Always start your hints with "Consider" or "Think about".
"""

