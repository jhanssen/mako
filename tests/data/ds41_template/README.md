# DeepSeek-V4.1 chat template cases

`case-N.txt` are DeepSeek's reference encoder outputs (`encoding/tests/test_output_N.txt` in the
deepseek-ai/DeepSeek-V4.1-Flash-1017 repository, revision 1bae0e1088993b9b82de08f59de5be5a34e42170), unchanged.
`case-N.json` are the matching `test_input_N.json` as template variables: the messages and tools, `thinking_mode`
(the fixture's, chat when it has none), `reasoning_effort` when given, `add_generation_prompt` true, and tool call
arguments given as JSON text turned into objects (as the server does).

Copyright (c) 2023 DeepSeek, MIT License.
