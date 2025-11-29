from numpy import array
from numpy.linalg import norm
from test_utils import tts, register_testing, run_testing

module = 'embedding'

@register_testing(module)
async def normal_test():
    text = 'hello world!'
    embed = (await tts.embedding(text=text))['embedding']
    return (
        f'Input text: {text}.\n'
        f'Embedding length: {len(embed)}.\n'
        f'Embedding: \n{embed[:5]}...{embed[-5:]}'
    )

@register_testing(module)
async def compare_test():
    query = '贵州和广西相比'
    candidates = [
        '那我还是感觉我们贵州牛逼',
        'hello world',
    ]
    query_emb = array((await tts.embedding(text=query))['embedding'])
    candidate_embeds = [array((await tts.embedding(text=c))['embedding']) for c in candidates]
    sims = [query_emb.dot(c_emb)/(norm(query_emb)*norm(c_emb)) for c_emb in candidate_embeds]
    return (
        f'query: `{query}`\n'
        'similarities:\n'
        + '\n'.join([f'  `{candidates[i]}`: {sims[i]:.4f}' for i in range(len(candidates))])
    )
    
    
if __name__ == '__main__':
    run_testing(module)
    