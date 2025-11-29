import os
import base64

from typing import Self
from pydantic import BaseModel, TypeAdapter

from test_utils import tts, register_testing, run_testing
from thinkthinksyn import google_search_tool, wiki_search_tool, LLM, Qwen3_30B_A3B_Omni_Instruct

module = 'completion'

@register_testing(module)
async def normal_test():
    return (await tts.completion(prompt='1+1? tell me ans directly without other words.'))['text'].strip()

@register_testing(module)
async def stream_test():
    text = ''
    async for chunk in tts.stream_completion(prompt='Count from 1 to 5, without other words.'):
        text += chunk['data'] if chunk['event'] == 'message' else ''
    return text.strip()

# @register_testing(module)
# async def internal_tool_test():
#     return (await tts.completion(
#         prompt='Who is zutomayo?.',
#         tools=[google_search_tool, wiki_search_tool],
#         tool_config={'call_mode': 'inline'}
#     ))['text']
    
@register_testing(module)
async def test_image():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(curr_dir, 'test-page.png')
    with open(img_path, 'rb') as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    class Paragraph(BaseModel):
        title: str|None = None
        content: str
        children: list[Self]|None = None
    
    adapter = TypeAdapter(list[Paragraph])
    schema = adapter.json_schema()
    prompt = f'这是一个历史教材的页面, 提取当中**所有**有意义的段落, 以以下的json schema 返回: \n{str(schema)}.\n<__media__>'
    result = await tts.completion(
        messages=[{'role': 'user', 'content': prompt, 'medias': {0: {'type': 'image', 'content': img_b64}}}],
        json_schema=schema,
        model_filter=(LLM.Name == Qwen3_30B_A3B_Omni_Instruct.Name),
    )
    return adapter.validate_json(result['text'])

@register_testing(module)
async def test():    
    class CalculationResult(BaseModel):
        explain: str
        result: float|int

    class MCResult[T](BaseModel):
        solution: T
        chosen: str

    r = (await tts.completion(
        prompt=f'Solve the math problem: 2+2*2. Here are the choices: A. 6, B. 8, C. 4. Respond with this format: {MCResult[CalculationResult].model_json_schema()}',
        json_schema=MCResult[CalculationResult].model_json_schema()
    ))['text']
    return MCResult[CalculationResult].model_validate_json(r)
    
if __name__ == '__main__':
    run_testing(module)