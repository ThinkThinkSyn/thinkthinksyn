import os
import base64

from typing import Self
from pydantic import BaseModel, TypeAdapter

from test_utils import tts, register_testing, run_testing
from thinkthinksyn import (google_search_tool, wiki_search_tool, LLM, Qwen3_30B_A3B_Omni_Instruct)

module = 'completion'

@register_testing(module)
async def simple_test():
    return (await tts.completion(prompt='1+1? tell me ans directly without other words.'))['text'].strip()

@register_testing(module)
async def streaming_test():
    text = ''
    async for chunk in tts.stream_completion(prompt='Count from 1 to 5, without other words.'):
        text += chunk['data'] if chunk['event'] == 'message' else ''
    return text.strip()
    
@register_testing(module)
async def image_input_test():
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
async def json_schema_test():    
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
    
@register_testing(module)
async def json_complete_test():
    class Person(BaseModel):
        name: str
        age: int
        friends: list[str]
    
    r = await tts.json_complete(
        'Generate a json object representing a person named Alice, age 30, with friends Bob and Charlie. Respond only with the json object.',
        return_type=Person,
    )
    return r

# @register_testing(module)
# async def internal_tool_test():
#     return (await tts.completion(
#         prompt='Who is zutomayo?.',
#         tools=[google_search_tool, wiki_search_tool],
#         tool_config={'call_mode': 'inline'}
#     ))['text']
    
if __name__ == '__main__':
    # run_testing(module)
    import re
    import asyncio
    from pydantic import Field
    
    async def test_math():
        
        class Method(BaseModel):
            abstract: str
            steps: list[str]
            
            def __repr__(self):
                return f'{self.abstract}\nSteps: \n{self.steps}'
        
        class Approaches(BaseModel):
            possible_methods: list[Method]
        
        img_p = r"C:\Users\MSI-NB\Desktop\PyProjs\thinkthinksyn\tmp\math-q.png"
        with open(img_p, 'rb') as f:
            img_data = f.read()
        
        async def plan(img_data)-> Approaches:
            prompt = r'''
            对于这个数学问题, 思考有什么可能的解题方法, 并列出每种方法的大致描述和抽象步骤.
            注意, 你不需要解题, 只需要列出你认为可能的解题方法. 方法的数量最多3种, 如果你认为只有1种方法, 就只列出1种. 
            '''
        
            r = await tts.completion(
                messages=[{
                    'role': 'system', 'content': prompt, 
                    'medias': {0: {'type': 'image', 'content': base64.b64encode(img_data).decode('utf-8')}}}
                ],
                model_filter=str(LLM.Name == Qwen3_30B_A3B_Omni_Instruct.Name),
                json_schema=Approaches.model_json_schema()
            )
            return Approaches.model_validate_json(r['text'])
        
        class Solve(BaseModel):
            think: str
            ans: str|None = Field(default=None, description='The final answer(if solved successfully)')
            
        async def solve(img_data, method: Method) -> Solve:
            prompt = f'''
            使用以下方法解答这个数学问题:
            方法描述: {method.abstract}
            方法步骤: {'; '.join(method.steps)}
            
            如果在思考过程中你认为这个方法无法解答这个问题, 你可以选择放弃. 这个情况下不需要给出答案; 反之, 请给出最终答案.
            注意: 
            1. 不允许用其他方法来解题, 只能尝试上述方法(有其他人会帮你尝试其他方法, 你的责任是专注于思考这个方案, 你思考别的方法反而是侮辱了其他人的努力)
            2. **不要**纠缠在某一步, 发现无法继续时, 选择放弃，大家都等你很久了, 别浪费大家时间
            
            思考部分请用<think></think>标签包裹, 答案部分请用<ans></ans>标签包裹. 
            '''
            o = ''
            try:
                last_count = 0
                last_output = '<think>'
                async for r in tts.stream_completion(
                    messages=[{
                        'role': 'system', 'content': prompt, 
                        'medias': {0: {'type': 'image', 'content': base64.b64encode(img_data).decode('utf-8')}}}
                    ],
                    model_filter=str(LLM.Name == Qwen3_30B_A3B_Omni_Instruct.Name),
                    prefix='<think>',
                    config={'max_tokens': 1024},
                ):
                    if r['event'] == 'message':
                        o += r['data']
                        last_count += 1
                        last_output += r['data']
                        if last_count >= 50:
                            print(f'Current output: \n{last_output}\n---')
                            last_count = 0
                            last_output = ''
                            
                think, ans = None, None
                think_m = re.search(r'<think>(.*?)</think>', o, re.DOTALL)
                ans_m = re.search(r'<ans>(.*?)</ans>', o, re.DOTALL)
                if think_m:
                    think = think_m.group(1).strip()
                if ans_m:
                    ans = ans_m.group(1).strip()
                return Solve(think=think or '', ans=ans)
            except Exception as e:
                print(f'Error during solving with method {method.abstract}: {e}')
                return Solve(think=f'解题过程中出现异常: {e}', ans=None)
            
        async def solve_problem(img_data) -> Solve|None:
            approaches = await plan(img_data)
            print(f'Planned approaches: {[repr(m) for m in approaches.possible_methods]}')
            solutions = await asyncio.gather(*[solve(img_data, m) for m in approaches.possible_methods])
            for i, sol in enumerate(solutions):
                if sol.ans is not None:
                    return sol
                else:
                    print(f'Method {approaches.possible_methods[i].abstract} failed to solve the problem.\nThoughts: {sol.think}')
            print(f'All methods failed to solve the problem.\nMethods tried: {[m.abstract for m in approaches.possible_methods]}')
            return None
        
        sol = await solve_problem(img_data)
        if not sol:
            print('无法解答此问题')
        else:
            print(f'问题解答成功!\n思考过程: {sol.think}\n最终答案: {sol.ans}')
    
    async def test_math_simple():
        img_p = r"C:\Users\MSI-NB\Desktop\PyProjs\thinkthinksyn\tmp\math-q.png"
        with open(img_p, 'rb') as f:
            img_data = f.read()
            
        async for r in tts.stream_completion(
            messages=[{
                'role': 'system', 'content': 'solve this problem. Return answer once you get it, no need dummy validations.', 
                'medias': {0: {'type': 'image', 'content': base64.b64encode(img_data).decode('utf-8')}}}
            ],
            model_filter=str(LLM.Name == Qwen3_30B_A3B_Omni_Instruct.Name),
        ):
            if r['event'] == 'message':
                print(r['data'], end='', flush=True)
    
    
    # asyncio.run(test_math())
    asyncio.run(test_math_simple())