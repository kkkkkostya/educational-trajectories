def read_file(path: str, encoding: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except PermissionError as e:
        raise PermissionError(f"Insufficient rights to open the file: {path}") from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end,
                                 f"Decoding error: possibly incorrect encoding in {path}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred {path}: {e}") from e