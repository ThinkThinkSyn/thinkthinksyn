import time
import aioftp
import base64
import aiohttp
import aiofiles
import fnmatch

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, ParseResult
from aioftp import DEFAULT_PASSWORD as FTP_DEFAULT_PASSWORD, DEFAULT_PORT as FTP_DEFAULT_PORT, DEFAULT_USER as FTP_DEFAULT_USER
from typing import (BinaryIO, Generator, AsyncGenerator, Sequence, Literal, AsyncIterable, Iterable, 
                    overload, TypeVar)
from typing_extensions import TypeAliasType

_T = TypeVar('_T')
_IO = TypeVar('_IO', bound=BinaryIO)
_Generator = TypeAliasType("_Generator", Generator[_T, None, None]|AsyncGenerator[_T, None]|AsyncIterable[_T]|Iterable[_T], type_params=(_T,))
AcceptableFileSource = TypeAliasType("AcceptableFileSource", str|bytes|Path|_Generator[bytes]|_Generator[str])

_DEFAULT_MAX_SIZE = 256 * 1024 * 1024  # 256 MB
_DEFAULT_TIMEOUT = 120  # seconds

async def _save_get_stream(
    stream: _Generator[bytes],
    out: _IO|None=None,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT
) -> _IO:
    total_size = 0
    if out is None:
        output = BytesIO()
    else:
        output = out

    start = time.time()
    if isinstance(stream, (Generator, Iterable)):
        for chunk in stream:
            total_size += len(chunk)
            if max_size is not None and total_size > max_size:
                raise ValueError("Data exceeds maximum allowed size")
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Data retrieval timed out")   
            output.write(chunk)
    else:
        async for chunk in stream:
            total_size += len(chunk)
            if max_size is not None and total_size > max_size:
                raise ValueError("Data exceeds maximum allowed size")
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Data retrieval timed out")
            output.write(chunk)
    
    if isinstance(output, BytesIO):
        output.seek(0)
    return output   # type: ignore

# region path
@overload
async def save_get_path(
    path: str|Path,
    out: None=None,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
)-> BytesIO:...

@overload
async def save_get_path(
    path: str|Path,
    out: _IO,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
)-> _IO:...

async def save_get_path(
    path: str|Path,
    out: BinaryIO|None=None,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
):
    '''
    Retrieve data from a local file path and save it to a binary output stream.
    Parameters:
        path (str|Path): The file path to read from.
        out (BinaryIO|None): The output binary stream to write data to. If None, a BytesIO stream is created.
        max_size (int|None): Maximum allowed size of the data to retrieve. Defaults to 128 MB.
        timeout (int|float|None): Maximum time in seconds to wait for data retrieval. Defaults to 32 seconds.
        whitelist_dirs (Sequence[str|Path]|None): List of allowed directory paths. If None, all directories are allowed.
                                                 Wildcards are supported.
        blacklist_dirs (Sequence[str|Path]|None): List of disallowed directory paths. If None, no directories are disallowed.
                                                 Wildcards are supported.
    
    Returns:
        IO: The output binary stream containing the retrieved data.
    '''
    path = Path(path).resolve()
    
    # Check whitelist/blacklist
    if whitelist_dirs is not None:
        if not any(fnmatch.fnmatch(str(path.parent), str(pattern)) for pattern in whitelist_dirs):
            raise ValueError("Directory not in whitelist")
    if blacklist_dirs is not None:
        if any(fnmatch.fnmatch(str(path.parent), str(pattern)) for pattern in blacklist_dirs):
            raise ValueError("Directory is in blacklist")
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    async def file_stream():
        async with aiofiles.open(path, 'rb') as f:
            while True:
                chunk = await f.read(1024)
                if not chunk:
                    break
                yield chunk
    
    return await _save_get_stream(
        file_stream(),
        out,
        max_size,
        timeout
    )
# endregion

# region url
@overload
async def save_get_url(
    url: str,
    out: None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
) -> BytesIO: ...

@overload
async def save_get_url(
    url: str,
    out: _IO, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
) -> _IO: ...

async def save_get_url(
    url: str,
    out: BinaryIO|None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
) -> BinaryIO:
    '''
    Retrieve data from a URL (HTTP or FTP) and save it to a binary output stream.
    Parameters:
        url (str): The URL to retrieve data from.
        out (BinaryIO|None): The output binary stream to write data to. If None, a BytesIO stream is created.
        max_size (int|None): Maximum allowed size of the data to retrieve. Defaults to 128 MB.
        timeout (int|float|None): Maximum time in seconds to wait for data retrieval. Defaults to 32 seconds.
        whitelist_domains (Sequence[str]|None): List of allowed domain names (e.g. thinkthinksyn.com). If None, all domains are allowed.
                                              Wildcards are supported.
        blacklist_domains (Sequence[str]|None): List of disallowed domain names. If None, no domains are disallowed.
                                                Wildcards are supported.
    
    Returns:
        IO: The output binary stream containing the retrieved data. Defaults to BytesIO if no output stream is provided.
    '''
    url = url.strip()
    def _parse(url: str) -> tuple[Literal['http', 'ftp'], str, ParseResult]:
        parsed = urlparse(url)
        assert parsed.scheme in ('http', 'https', 'ftp', 'ftps'), "Unsupported URL scheme"
        host = parsed.netloc
        if not host and parsed.path:
            host = parsed.path.split('/')[0]
        if parsed.scheme in ('http', 'https'):
            return 'http', host, parsed
        else:
            return 'ftp', host, parsed

    protocol, host, parsed = _parse(url)
    if whitelist_domains is not None:
        if not any(fnmatch.fnmatch(host, pattern) for pattern in whitelist_domains):
            raise ValueError("Domain not in whitelist")
    if blacklist_domains is not None:
        if any(fnmatch.fnmatch(host, pattern) for pattern in blacklist_domains):
            raise ValueError("Domain is in blacklist")

    if protocol == 'http':
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to retrieve URL: {response.status}")
                async def http_stream():
                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk
                return await _save_get_stream(
                    http_stream(),
                    out,
                    max_size,
                    timeout
                )
    else:
        ftp_host = parsed.hostname or host
        ftp_port = parsed.port or FTP_DEFAULT_PORT
        ftp_user = parsed.username or FTP_DEFAULT_USER
        ftp_pass = parsed.password or FTP_DEFAULT_PASSWORD
        ftp_path = parsed.path or '/'
        async with aioftp.Client.context(
            ftp_host,
            ftp_port,
            ftp_user,
            ftp_pass
        ) as client:
            stream = await client.download_stream(ftp_path)
            async def ftp_stream():
                async for chunk in stream.iter_by_block(1024):
                    yield chunk
            return await _save_get_stream(
                ftp_stream(),
                out,
                max_size,
                timeout
            )
# endregion

# region base64
@overload
async def save_get_base64(
    b64_string: str|_Generator[str],
    out: None=None,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> BytesIO: ...

@overload
async def save_get_base64(
    b64_string: str|_Generator[str],
    out: _IO,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> _IO: ...

async def save_get_base64(
    b64_string: str|_Generator[str], 
    out: BinaryIO|None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> BinaryIO:
    '''
    Decode base64 data and save it to a binary output stream.
    Parameters:
        b64_string (str|Generator[str]): The base64 string or generator of base64 strings.
        out (BinaryIO|None): The output binary stream to write data to. If None, a BytesIO stream is created.
        max_size (int|None): Maximum allowed size of the decoded data. Defaults to 128 MB.
        timeout (int|float|None): Maximum time in seconds to wait for decoding. Defaults to 32 seconds.

    Returns:
        IO: The output binary stream containing the decoded data.
    '''
    assert isinstance(b64_string, (str, Generator, AsyncGenerator, AsyncIterable, Iterable)), \
        "b64_string must be str or generator"
    if isinstance(b64_string, str):
        if not b64_string:
            if out is None:
                return BytesIO()
            return out
        if b64_string.startswith("data:"):
            b64_string = b64_string.split("base64,", 1)[1]
        async def b64_stream(): # type: ignore
            try:
                decoded = base64.b64decode(b64_string)
                yield decoded
            except Exception as e:
                raise ValueError(f"Invalid base64 data: {e}")
    else:
        # Generator of strings
        async def b64_stream():
            is_first = True
            buf = ''
            if isinstance(b64_string, (Generator, Iterable)):
                for chunk in b64_string:
                    if is_first:
                        is_first = False
                        if chunk.startswith("data:"):
                            chunk = chunk.split("base64,", 1)[1]
                            if not chunk:
                                continue
                    buf += chunk
                    to_be_decoded, buf = buf[:-((len(buf)) % 4)], buf[-((len(buf)) % 4):]
                    try:
                        decoded = base64.b64decode(to_be_decoded)
                        yield decoded
                    except Exception as e:
                        raise ValueError(f"Invalid base64 data: {e}")
            else:
                async for chunk in b64_string:
                    if is_first:
                        is_first = False
                        if chunk.startswith("data:"):
                            chunk = chunk.split("base64,", 1)[1]
                            if not chunk:
                                continue
                    buf += chunk
                    to_be_decoded, buf = buf[:-((len(buf)) % 4)], buf[-((len(buf)) % 4):]
                    try:
                        decoded = base64.b64decode(to_be_decoded)
                        yield decoded
                    except Exception as e:
                        raise ValueError(f"Invalid base64 data: {e}")
            if buf:
                try:
                    decoded = base64.b64decode(buf)
                    yield decoded
                except Exception as e:
                    raise ValueError(f"Invalid base64 data: {e}")
    
    return await _save_get_stream(
        b64_stream(),
        out,
        max_size,
        timeout
    )
# endregion

# region bytes
@overload
async def save_get_bytes(
    byte_data: bytes|_Generator[bytes], 
    out: None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> BytesIO: ...

@overload
async def save_get_bytes(
    byte_data: bytes|_Generator[bytes],
    out: _IO,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> _IO: ...

async def save_get_bytes(
    byte_data: bytes|_Generator[bytes], 
    out: BinaryIO|None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
) -> BinaryIO:
    '''
    Process byte data and save it to a binary output stream.
    Parameters:
        byte_data (bytes|Generator[bytes]): The byte data or generator of byte chunks.
        out (BinaryIO|None): The output binary stream to write data to. If None, a BytesIO stream is created.
        max_size (int|None): Maximum allowed size of the data. Defaults to 128 MB.
        timeout (int|float|None): Maximum time in seconds to wait for processing. Defaults to 32 seconds.
    
    Returns:
        IO: The output binary stream containing the byte data.
    '''
    assert isinstance(byte_data, (bytes, Generator, AsyncGenerator, AsyncIterable, Iterable)), \
        "byte_data must be bytes or generator"
    if isinstance(byte_data, bytes):
        # Single bytes object
        async def _bytes_stream():
            yield byte_data
        bytes_stream = _bytes_stream()
    else:
        # Generator of bytes
        bytes_stream = byte_data    # type: ignore
    
    return await _save_get_stream(
        bytes_stream,   # type: ignore
        out,
        max_size,
        timeout
    )
# endregion

# region all-in-one
@overload
async def save_get(
    source: AcceptableFileSource, 
    out: None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    /,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
) -> BytesIO: ...

@overload
async def save_get(
    source: AcceptableFileSource,
    out: _IO,
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    /,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
) -> _IO: ...

async def save_get(
    source: AcceptableFileSource, 
    out: BinaryIO|None=None, 
    max_size: int|None=_DEFAULT_MAX_SIZE,
    timeout: int|float|None=_DEFAULT_TIMEOUT,
    /,
    whitelist_dirs: Sequence[str|Path]|None=None,
    blacklist_dirs: Sequence[str|Path]|None=None,
    whitelist_domains: Sequence[str]|None=None,
    blacklist_domains: Sequence[str]|None=None,
):
    '''
    Universal interface to retrieve data from various sources and save to a binary output stream.
    Automatically detects the source type and calls the appropriate method.
    
    Parameters:
        source: The data source - can be a URL string, file path, bytes, or generator.
        out (BinaryIO|None): The output binary stream to write data to. If None, a BytesIO stream is created.
        max_size (int|None): Maximum allowed size of the data. Defaults to 128 MB.
        timeout (int|float|None): Maximum time in seconds to wait for processing. Defaults to 32 seconds.
        whitelist_dirs (Sequence[str|Path]|None): List of allowed directory paths for file sources. If None, all directories are allowed.
                                                    Wildcards are supported.
        blacklist_dirs (Sequence[str|Path]|None): List of disallowed directory paths for file sources. If None, no directories are disallowed.
                                                    Wildcards are supported.        
        whitelist_domains (Sequence[str]|None): List of allowed domain names for URL sources. If None, all domains are allowed.
                                                Wildcards are supported.                        
        blacklist_domains (Sequence[str]|None): List of disallowed domain names for URL sources. If None, no domains are disallowed.
                                                Wildcards are supported.                
    Returns:
        BinaryIO: The output binary stream containing the retrieved data.
    '''
    if isinstance(source, bytes):
        return await save_get_bytes(source, out, max_size, timeout)
    elif isinstance(source, str):
        if source.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
            return await save_get_url(source, out, max_size, timeout, 
                                      whitelist_domains=whitelist_domains, blacklist_domains=blacklist_domains)
        elif source.startswith('data:') and ';base64,' in source[:64]:
            b64_data = source.split('base64,', 1)[1]
            return await save_get_base64(b64_data, out, max_size, timeout)
        elif ('/' not in source) and ('\\' not in source) and len(source) % 4 == 0:
            try:
                base64.b64decode(source, validate=True)
                return await save_get_base64(source, out, max_size, timeout)
            except:
                pass
        else:
            # assume it's a file path
            source_path = Path(source)
            if not source_path.exists():
                raise ValueError("Unknown source type or file path too long (>260 characters)")
        return await save_get_path(source_path, out, max_size, timeout, 
                                   whitelist_dirs=whitelist_dirs, blacklist_dirs=blacklist_dirs)
    elif isinstance(source, Path):
        return await save_get_path(source, out, max_size, timeout, 
                                   whitelist_dirs=whitelist_dirs, blacklist_dirs=blacklist_dirs)
    elif isinstance(source, (Generator, AsyncGenerator, AsyncIterable, Iterable)):
        try:
            if isinstance(source, Generator):
                first_item = next(source)
            elif isinstance(source, Iterable):
                first_item = next(iter(source))
            elif isinstance(source, AsyncGenerator):
                first_item = await source.__anext__()
            else: # AsyncIterable
                source = source.__aiter__()
                first_item = await source.__anext__()
        except (StopIteration, StopAsyncIteration):
            if out is None:
                return BytesIO()
            return out
        if isinstance(first_item, bytes):
            # Reconstruct generator with first item
            async def byte_gen():
                yield first_item  # type: ignore
                if isinstance(source, (Generator, Iterable)):
                    for item in source:  # type: ignore
                        yield item  # type: ignore
                else:
                    async for item in source:  # type: ignore
                        yield item  # type: ignore
            return await save_get_bytes(byte_gen(), out, max_size, timeout) # type: ignore
        else:
            # Reconstruct generator with first item
            async def str_gen():
                yield first_item  # type: ignore
                if isinstance(source, (Generator, Iterable)):
                    for item in source:  # type: ignore
                        yield item  # type: ignore
                else:
                    async for item in source:  # type: ignore
                        yield item  # type: ignore
            return await save_get_base64(str_gen(), out, max_size, timeout) # type: ignore
  
    raise ValueError(f"Got Unsupported source type: {type(source)}. Expected str (URL or path), bytes, Path, generator, or MinioPath.")
# endregion


__all__ = [
    "save_get_path",
    "save_get_url",
    "save_get_base64", 
    "save_get_bytes",
    "save_get",
    'AcceptableFileSource',
]

if __name__ == "__main__":
    import asyncio

    async def test():
        data = await save_get('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg')
        print(f"Retrieved data size: {len(data.read())} bytes")

        data = await save_get('https://api.thinkthinksyn.com/resources/tts/ab_asr_address_yue.wav')
        print(f"Retrieved data size: {len(data.read())} bytes")

    asyncio.run(test())