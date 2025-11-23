from test_utils import tts, run_testing, register_testing
from thinkthinksyn.data_types import google_search_tool, wiki_search_tool

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

@register_testing(module)
async def internal_tool_test():
    return (await tts.completion(
        prompt='Who is zutomayo?.',
        tools=[google_search_tool, wiki_search_tool],
        tool_config={'call_mode': 'inline'}
    ))['text']


if __name__ == '__main__':
    run_testing(module)