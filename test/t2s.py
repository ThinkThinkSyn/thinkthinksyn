import os
import base64
import wave

from test_utils import tts, register_testing, run_testing

_tmp_path = os.path.join(os.path.dirname(__file__), '..', 'tmp')
_tmp_path = os.path.abspath(_tmp_path)

module = 't2s'

@register_testing(module)
async def normal_test():
    text = 'hello world!'
    result = (await tts.t2s(text=text))
    audio_data = base64.b64decode(result['data'])
    with open(os.path.join(_tmp_path, 'normal_t2s_test.wav'), 'wb') as f:
        f.write(audio_data)
    return (
        f'result data length: {len(audio_data)}\n'
    )

@register_testing(module)
async def stream_test():
    text = 'hello world! ' * 2
    audio_data = b''
    channels = 1
    framerate = 24000
    sample_width = 2
    async for chunk in tts.stream_t2s(text=text):
        audio_data += base64.b64decode(chunk['data'])
        data_log = chunk.copy()
        data_log.pop('data')
        data_log['data'] = f'<{len(audio_data)} bytes>' # type: ignore
        print(f'Recv: {data_log}')
        channels = chunk.get('channels', channels)
        framerate = chunk.get('sampling_rate', framerate)
        sample_width = chunk.get('sampling_width', sample_width)
    
    with wave.open(os.path.join(_tmp_path, 'stream_t2s_test.wav'), 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(framerate)
        f.writeframes(audio_data)
        
    return (
        f'result data length: {len(audio_data)}\n'
    )
    
    
if __name__ == '__main__':
    run_testing(module)
    