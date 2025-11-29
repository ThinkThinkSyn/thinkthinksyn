from typing import Any, TypedDict, Literal, TypeAlias, Sequence
from typing_extensions import TypeAliasType, NotRequired

from .base import AIInput, AIOutput, JsonSchema
from .llm_tools import LLMTool

ChatMsgMediaType: TypeAlias = Literal["image", "audio", "video"]

class ChatMsgMedia(TypedDict, total=False):
    """
    A media content in the chat msg.
    NOTE:
        - each media is associated with a tag `<__MEDIA_{idx}__>` (or <__media__> <__MEDIA__>... is also acceptable) in the content.
        - 1 msg can have multiple media contents, e.g. image, audio, ...
    """

    type: ChatMsgMediaType
    """type of the media content type"""
    content: str
    """Raw content of the media msg. It could be a URL/ file path/ base64, ...."""
    textified_content: str|None
    """
    The textified content, i.e. image/audio is translated into meaningful text for
    inputting to non-multimodal LLM.
    
    This value will only be set when `get_text_content` is called. 
    WARNING: You should not pass this value directly except you know what you are doing.
    """

    # type specific configs
    use_audio_in_video: bool
    '''whether to use audio when extracting meaning from video.
    This field is only valid when `type` is `video`.'''

ChatMsgMedias = TypeAliasType("ChatMsgMedias", dict[int, ChatMsgMedia])

class ChatMsg(TypedDict, total=False):
    """Single chat msg with no role."""

    content: str
    """
    Raw content of the chat msg.
    You can include media label within text content to identify the location,
    e.g. `<__MEDIA_0__>` to refer to the media with id=0.
    
    Also, it is allowed for you to pass a dict like `{'content': 'data:image/png;base64,...', 'type': 'image'}`.
    In that case, it will be turned into:
    ```
    {
        'content': '<__MEDIA_0__>',
        'medias': {
            0: {'type': 'image', 'content': 'data:image/png;base64,...'}
        }
    }
    
    You can also use <__media__>/<__MEDIA__> without index. In that case, the order of media contents 
    in `medias` will be used to identify the location.
    ```
    WARNING: Each media can at most have 1 tag in the content. Duplicate tags will be escaped.
    """
    textified_content: str|None
    """
    The textified content, i.e. image/audio is translated into meaningful text for
    inputting to non-multimodal LLM.
    
    This value will only be set when `get_text_content` is called. 
    WARNING: You should not pass this value directly except you know what you are doing.
    """
    timestamp: int
    """
    Timestamp this the chat msg. Default to be current time.
    This field will only be used in some special cases, e.g. packing as a conversation prompt.
    By default, it will not be used in normal LLM completion. 
    NOTE: the value should in ms, i.e. 13 digits.
    """
    multi_modal: bool|None
    """
    Whether passing media contents directly to multi-modal model(if available).
    This field will override `LLMInput.multi_modal` field.
    """
    medias: ChatMsgMedias
    '''media contents included in this chat msg. Key is media index.'''

class ChatMsgWithRole(ChatMsg):
    role: str

SingleLogitBias = TypeAliasType("SingleLogitBias", tuple[str|int, float|bool])
'''
Single logit bias, e.g. ("token", 0.5) or ("token", True) for enable/disable token.
For number case, the range is [-100, 100] 
'''

class CompletionConfig(TypedDict, total=False):
    """
    Config for completion generation. Compatible with OpenAI format.
    As most of nodes are using llama.cpp as inference engine,
    for more config details, you can refer to https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
    """

    max_tokens: int
    """
    How many tokens to generate at most(also called `max_new_token`).
    This value will be adjusted automatically to fit the model's max token limit
    during `LLM.validate_input` method.
    """
    temperature: float | None
    """
    (0.0, 2.0], higher->more random, lower->more deterministic.
    Default is 0.7, which is the same as OpenAI.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation.
    """
    
    frequency_penalty: float|None
    """
    Repeat alpha frequency penalty.
    [-2.0, 2.0], higher->more repetition, lower->less repetition.
    For difference between `presence_penalty` and `frequency_penalty`,
    see: https://blog.csdn.net/jarodyv/article/details/129062982
    """
    presence_penalty: float|None
    '''
    Repeat alpha presence penalty. >0 means a higher penalty discourages the 
    model from repeating tokens
    For difference between `presence_penalty` and `frequency_penalty`,
    see: https://blog.csdn.net/jarodyv/article/details/129062982
    '''
    repetition_penalty: float | None
    """
    Control the repetition of token sequences in the generated text.
    [1.0, +inf), higher->less repetition, lower->more repetition
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation.
    """
    
    top_k: int | None
    """[1, +inf), number of top tokens to choose from.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation."""
    top_p: float | None
    """[0.0, 1.0], cumulative probability of top tokens to choose from.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation."""
    min_p: float | None
    """[0.0, 1.0], minimum cumulative probability of top tokens to choose from"""
    
    stop: str | list[str] | tuple[str, ...]
    """stop token(s) for generation. If None, it will use the default stop token of the model."""
    ignore_eos: bool|None
    '''ignore end of stream token and continue generating.
    WARN: this may result in infinite generation, so use with caution.'''
    logit_bias: list[SingleLogitBias]|None
    """
    Modify the likelihood of a token appearing in the generated text completion. 
    For example, use "logit_bias": [[15043,1.0]] to increase the likelihood of the token 'Hello', 
    or "logit_bias": [[15043,-1.0]] to decrease its likelihood. Setting the value to false, 
    "logit_bias": [[15043,false]] ensures that the token Hello is never produced. 
    The tokens can also be represented as strings, e.g. [["Hello, World!",-0.5]] will reduce the
    likelihood of all the individual tokens that represent the string Hello, World!, 
    just like the presence_penalty does. 
    """

ToolChoiceMode: TypeAlias = Literal["none", "auto", "required", "required_one"]
"""
Equivalent to `tool_choice` in OpenAI(`required_one` is our extra options):
Modes:
    - `none`: No tool will be chosen. In this case, message will be returned instead of tool calling. 
    - `auto`: Automatically choose 1 or more tools, or not choosing any.
    - `required`: One or more tools must be chosen.
    - `required_one`: must choose 1 tool.
"""
ToolCallMode: TypeAlias = Literal["define", "call", "inline"]
"""
Mode of tool call:
- `define`: Define the input params of the tool(s) should be called. This is the default mode when the tool
            is chosen, which is the same as OpenAI.
- `call`: (For internal tools only) After defining the input params, call the tool(s) with the defined params
          and get the result from `value` field. Note that this is only available when the tool is found 
          within backend server (by inheriting the `LLMTool` class).
- `inline`: (For internal tools only) After calling the tool, the calling result will be built as prompt and
            pass into LLM again to enforce its ability answering user's question. This is the default mode in 
            `chat` service.
"""

class ToolConfig(TypedDict, total=False):
    """configs for `tools` in LLMInput"""

    choose_mode: ToolChoiceMode
    """
    Equivalent to `tool_choice` in OpenAI(`required_one` is our extra options):
    Modes:
        - `none`: No tool will be chosen. In this case, message will be returned instead of tool calling. 
        - `auto`: Automatically choose 1 or more tools, or not choosing any. If a tool should be called multiple times,
                  it will also be chosen with multiple params.
        - `required`: One or more tools must be chosen.
        - `required_one`: must choose 1 tool.
    NOTE: this field will be ignored if `tool_force_chosen` is set.
    """
    tool_force_chosen: str | list[str] | None
    """
    Manually set the final chosen tool. It should be a list of tool names or a single tool name.
    Your manual choice should be in the list of `tools` in `LLMInput`.
    Note: In OpenAI's format, this field is also put under `tool_choice`. 
          In this case, the value will be redirected to `tool_chosen`.
    """
    call_mode: ToolCallMode
    """
    Mode of tool call (default to be `define`):
    - `define`: Define the input params for calling the tool(s).
                This is the default mode, which is the same as OpenAI.
    - `call`: Call the tool(s) with params given by LLM finally, and get the result from 
              `value` field of `ToolCallResult`. Note that this is only available 
              when the tool is found within backend server (by inheriting the 
              `LLMTool` class).
              3rd-party-tool-upload system will be supported in the future.
    - `inline`: After calling the tool, the calling result will be built as prompt and pass 
                into LLM again to enforce its ability answering user's question.
                This is the default mode in `chat` service.   
    """

ToolParamType: TypeAlias = Literal['required', 'optional', 'hidden', 'return']
'''Type of the tool parameter. `optional` is only available when the parameter has a default value.'''

class ToolParam(JsonSchema, total=False):
    param_type: ToolParamType
    '''Type of the parameter. Note: `optional` is only available when the parameter has a default value.'''

class ToolInfo(TypedDict):
    '''
    Information about a LLM tool, for sending to model. Compatible with OpenAI's `tools` field.
    You should fill detailed information to get better result in calling. 
    '''
    
    name: str
    '''name of the tool. This is an important field for meaningful selection.'''
    description: NotRequired[str|None]
    '''description of the tool. This is an important field for meaningful selection.'''
    internal_key: NotRequired[str|None]
    '''For internal tools only, a unique string. See `llm_tools` module for more details.'''    
    params: dict[str, ToolParam]
    '''
    parameters of the tool. Each key is the name of the parameter, and the value is the detail of the parameter.
    Tool params should follows the specification of json schema. See: https://json-schema.org/understanding-json-schema
    '''
    return_type: NotRequired[ToolParam|None]
    '''Return type of the tool, for adding extra information to LLM.'''
    tool_creation_params: NotRequired[dict[str, Any]]
    '''
    Parameters for creating the tool instance, e.g. API key, model name, etc.
    This field is only available for internal tools, and when mode in `call` or `inline`.
    '''

_ToolType = TypeAliasType("_ToolType", str|ToolInfo|LLMTool)

class CompletionInput(AIInput, total=False):
    '''Input for LLM completion.'''
    
    context_id: str
    '''
    The context id for this input.
    If not given, a random context id will be generated.
    '''
    prompt: str
    '''
    The prompt to pass to the model. 
    Note:
        - If both `prompt` and `history`(`message`) are given, `prompt` will be act 
            as the final user input and `history` will be act as the chat history.
        - If `send_prompt_directly` is True, `prompt` will be send to model directly. No modification will be made.
    '''
    prefix: str
    '''
    The prefix before generation(or say the suffix after the built prompt). 
    It is helpful for special usage, e.g. adding `{` as prefix will make LLM more likely to output a perfect json.

    Note: when `tools` are given, this field will be ignored.
    '''
    system_prompt: str|None
    '''
    the system prompt to pass to the model. 
    You can still manually add system prompt in `history` by `system` role.
    '''
    messages: list[ChatMsgWithRole]
    '''
    The chat messages. It is used for inputting the full chat history, e.g.
    ```[{"role": "user", "content": "Hello!"}, ...]```
    NOTE: If your prompt has been included in this field, then no need to enter in field `prompt` anymore.
    '''
    send_prompt_directly: bool
    '''
    When send_prompt_directly=True, `prompt` field will not be modified(i.e. history & system_prompt will be ignore.), 
    and it will be send to model directly.
    '''
    completion_start_role: str
    '''
    When building prompt, it will be ended with a role's start tag, e.g. 
    ```
    {"role": "user", "content": "Hello!"} -> "user: Hello!<end_of_input_token>model: "
    ```
    You can change such starting role by this field. This will be helpful when you are
    doing some special completions, e.g. jail breaking by role inversion.
    This is field only available when `send_prompt_directly` is False.
    '''
    config: CompletionConfig
    '''The config for generating completion.'''
    openai_compatible: bool
    '''
    Whether to return a openai compatible output.
    DEPRECATED WARNING: this field hasn't being maintained for a long time. Not recommended to use anymore.
    USAGE WARNING: when this field is True, the returned output will not be `CompletionOutput` anymore.
    '''
    tools: _ToolType|Sequence[_ToolType]|None
    '''
    The tools to use in this input. Each tool should be a `LLMToolInfo` object,
    and each param in tool should be described as JSON schema.
    You can also pass tool key directly for internal tools.
    
    Note: tools will be ignored when:
        - if `send_prompt_directly` is True.
        - if `stream`=True
        - if `tool_config.mode`==`none`.
    '''
    tool_config: ToolConfig
    '''
    Config for tool calling. This is only available when `tools` is given.
    Tool setting fields in OpenAI's format(e.g. tool_choice, ...) when be put to this config automatically
    if you don't specify them in `tool_config` field.
    '''
    json_schema: dict[str, Any]|None
    '''
    Json schema restricting the model to reply.
    This is only available when there is any json-schema available nodes, e.g. llama.cpp.
    Otherwise, this field will be ignored.
    
    NOTE: `json_schema` usually just limit the response in a certain structure,
        i.e. you still need to prompt the model to output in that json format.
    '''
    multi_modal: bool
    '''
    This field determines whether to allow building multimodal input/output prompts.
    E.g. when a multi-modal model is deployed under a multi-modal supporting framework,
    when `multi_modal` is True, image/audio messages will not be passed to Img2Text/S2T service
    anymore, but will be passed to the model directly.
    
    If you wanna ensure using multi-modal model, you still need to pass `model_filter`, `model` or `node_filter`
    to make sure you will select a multi-modal node.
    '''

class ToolCallResult(TypedDict):
    '''result of the tool call. If mode=`call`, `value` will be attached to the return.'''
    
    name: str
    '''name of the tool. Note that your chosen tool must be included in `tools` field of `LLMInput`.'''
    params: dict[str, Any]
    '''parameters for calling the tool, determined by LLM. {param_name, value}'''
    internal_key: str|None
    '''internal id for the tool. This value only exists when the tool is registered within thinkthinksyn backend.'''
    
    # for `call`/`inline` mode
    value: Any
    '''
    Result of the tool call. This field is only available when mode=`call`/`inline`.
    If tool is not found or running process is failed, error msg will be attached to `error` field.
    '''
    error: str|None
    '''
    If the calling process is failed, the error message will be attached here.
    This is only for mode=`call`/`inline`.
    '''

class CompletionOutput(AIOutput[CompletionInput]):
    text: str
    '''
    The final output text of the LLM. Note that the text is concatenated with 
    your given `prefix` in `CompletionInput`.
    '''
    input_token_count: int
    '''The number of tokens in the input prompt.'''
    output_token_count: int
    '''The number of tokens in the output completion.'''
    tool_call_results: list[ToolCallResult]|None
    '''result of tool callings(only when tools are used).'''

# region stream output types
class CompletionMessageStreamOutput(TypedDict):
    '''A chunk of completion message stream output.'''
    event: Literal["message"]
    '''The event type.'''
    data: str
    '''The delta text generated in this chunk.'''

class CompletionToolCallStreamOutput(TypedDict):
    '''A tool call request in stream output.'''
    event: Literal["tool_calling"]
    '''The event type.'''
    data: list[ToolCallResult]
    '''The tool call request data.'''

CompletionStreamOutput = TypeAliasType(
    "CompletionStreamOutput", 
    CompletionMessageStreamOutput | CompletionToolCallStreamOutput
) 
# endregion


__all__ = [
    'ChatMsgMediaType',
    'ChatMsgMedia',
    'ChatMsgMedias',
    'ChatMsg',
    'ChatMsgWithRole',
    
    'SingleLogitBias',
    'CompletionConfig',
    
    'ToolChoiceMode',
    'ToolParamType',
    'ToolParam',
    'ToolCallMode',
    'ToolConfig',
    'ToolInfo',
    'ToolCallResult',
    
    'CompletionInput',
    'CompletionOutput',

    'CompletionStreamOutput',
    'CompletionMessageStreamOutput',
    'CompletionToolCallStreamOutput',
]