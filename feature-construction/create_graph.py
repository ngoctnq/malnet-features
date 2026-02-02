import os
import re
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from androguard.misc import AnalyzeAPK
from androguard.core.dex import EncodedMethod
from androguard.core.analysis.analysis import Analysis, ExternalMethod
from sklearn.feature_extraction import FeatureHasher
from typing import List, Tuple
from requests import post


# hyperparameters for entropy calculation
window = 128  # small because average method size is small
step = window // 2

global port
port = 8000  # default port for the inference server


def create_fcg_graph(dex_path: str) -> None:
    """
    Extract function call graph from APK file.

    Save file format: a dictionary consisting of:
        save_content['node_features']: numpy.ndarray[shape=(num_nodes, num_features)]
            the features of each EncodedMethod node
        save_content['edge_index']: Set[(from_node, to_node)]
            the edge list between only EncodedMethod nodes
        save_content['external_names']: Dict[int, str]
            mapping from external node index to the name of the external method
        save_content['edge_index_v2']: Set[(from_node, to_node)]
            the edge list between EncodedMethod and ExternalMethod nodes

    :param dex_path: path to APK file
    """

    # Disable Androguard logger inside fork
    from androguard.core.androconf import logger

    logger.disable("androguard")

    save_path = dex_path.replace(".apk", ".pkl")
    # glob returns the full path, so the "middle" token always exists
    save_path = save_path.rsplit("/", 2)
    save_path[1] += "-features"
    save_path = "/".join(save_path)

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_content = {}

        try:
            # print("[*]", dex_path)
            a, d, dx = AnalyzeAPK(dex_path)
            cg = dx.get_call_graph()

            node_index = 0
            node_mapping = {}

            nodes = list(cg.nodes())
            for node in nodes:
                node_mapping[node] = node_index
                node_index += 1

            edge_index = set()
            for n1, n2 in cg.edges():
                n1 = node_mapping[n1]
                n2 = node_mapping[n2]
                edge_index.add((n1, n2))

            save_content["node_features"] = [
                extract_features(node, dx) for node in nodes
            ]

            save_content["edge_index"] = edge_index
            save_content["external_names"] = dict(
                (node_mapping[node], get_ext_node_name(node, dx))
                for node in nodes
                if isinstance(node, ExternalMethod)
            )

            with open(save_path, "wb") as f:
                pickle.dump(save_content, f)

        except Exception as e:
            print("[!]", dex_path)
            print(e)
            raise e


def get_name_features(node, dx: Analysis) -> np.ndarray:
    """
    Extract features from a method node's class name and method name.

    :param node: method node
    :param dx: Analysis
    :return: numpy.ndarray[shape=(100,)]
    """
    all_features = []
    if isinstance(node, EncodedMethod):
        # class name -- e.g. Lcom/example/MyClass;
        class_name = node.get_class_name()
        # method name -- e.g. myMethod
        method_name = node.get_name()
    else:
        # ExternalMethod
        class_name, method_name = get_ext_node_name(node, dx)

    assert class_name.startswith("L") and class_name.endswith(";")
    # bag-of-word, but with each token, case-sensitive, separated by '/'
    class_name_tokens = class_name[1:-1].split("/")
    all_features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([class_name_tokens])  # type: ignore
        .toarray()[0]  # type: ignore
    )
    all_features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([[method_name]])  # type: ignore
        .toarray()[0]  # type: ignore
    )

    return np.concatenate(all_features)


def get_ext_node_name(node: ExternalMethod, dx: Analysis) -> Tuple[str, str]:
    """
    Get the name of the ExternalMethod node.

    :param node: ExternalMethod node
    :param ds: list of DEX files
    :return class_name: name of the class "containing" the method
    :return method_name: name of the method
    """

    class_name = node.get_class_name()
    method_name = node.get_name()
    clazz = dx.get_class_analysis(class_name)

    while True:
        # assumption is that the method is defined in the last superclass
        if clazz is None or clazz.is_external():
            return class_name, method_name

        # check if defined in this class
        class_def = clazz.get_class()
        for method in class_def.get_methods():
            if method.get_name() == method_name:  # type: ignore
                return class_name, method_name

        # otherwise, check the superclass
        class_name = clazz.get_class().get_superclassname()  # type: ignore
        clazz = dx.get_class_analysis(class_name)


def get_code_features(node: EncodedMethod, window: int, step: int) -> np.ndarray:
    """
    Extract features from EncodedMethod node's code section.

    The features are:
    - Code length
    - Byte histogram
    - Byte entropy histogram
    - Source code embedding

    :param node: EncodedMethod node
    :param window: window size for entropy calculation
    :param step: step size for entropy calculation
    :return: numpy.ndarray[shape=(50,)]
    """

    # statistics of the raw code
    raw_code = node.get_code().get_raw()  # type: ignore

    code_features = [
        np.array(
            [
                node.get_length(),
                len(raw_code),
            ]
        )
    ]

    raw_code = np.frombuffer(raw_code, dtype=np.uint8)
    # byte histogram
    byte_histogram = np.histogram(raw_code, bins=256, range=(0, 256))[0]
    code_features.append(byte_histogram / (byte_histogram.sum() + 1e-12))

    # byte entropy histogram
    output = np.zeros((16, 16), dtype=int)
    # calculate entropy for each window, at least 1 if raw code size is smaller than window
    start_idx_max = max(len(raw_code) - window + 1, 1)
    # strided trick is meaningless as we use a for loop anyway
    for start_idx in range(0, start_idx_max, step):
        block = raw_code[start_idx : start_idx + window]
        Hbin, c = _entropy_bin_counts(block, window)
        output[Hbin, :] += c

    output = output.flatten().astype(np.float32)
    code_features.append(output / (output.sum() + 1e-12))

    # code embedding
    try:
        source_code = node.get_source()
        if not source_code:
            raise AttributeError("No source code found")
    except (AttributeError, KeyError, IndexError):
        pass
    else:
        while True:
            req = post(
                f"http://localhost:{port}/",
                json={"code": source_code},
                headers={"Content-Type": "application/json"},
                timeout=None,
            )
            if req.status_code == 200:
                break
        embedding = np.frombuffer(req.content, dtype=np.float32)
        code_features.append(embedding)

    return np.concatenate(code_features, axis=0)


def _entropy_bin_counts(block: np.ndarray, window: int) -> Tuple[int, np.ndarray]:
    """
    Compute the entropy of a block and return the bin counts of the entropy histogram.
    Based on the EMBER feature extractor.

    :param block: the block to compute the entropy on
    :param window: the window size
    :return: tuple of entropy bin and bin counts
    """
    # coarse histogram, 16 bytes per bin
    c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
    p = c.astype(np.float32) / window
    wh = np.where(c)[0]
    # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)
    H = np.sum(-p[wh] * np.log2(p[wh])) * 2

    Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
    if Hbin == 16:  # handle entropy = 8.0 bits
        Hbin = 15

    return Hbin, c


def get_access_flag_features(node: EncodedMethod) -> np.ndarray:
    """
    Extract features from EncodedMethod node's access flags.

    For reference:
    public:       0x1
    private:      0x2
    protected:    0x4
    static:       0x8
    final:        0x10
    synchronized: 0x20
    bridge:       0x40
    varargs:      0x80
    native:       0x100
    interface:    0x200
    abstract:     0x400
    strictfp:     0x800
    synthetic:    0x1000
    constructor:  0x10000

    :param node: EncodedMethod node
    :return: numpy.ndarray[shape=(14,)]
    """
    features = []
    access_flags = node.get_access_flags()
    features.append((access_flags >> 16) & 1)  # constructor
    for i in range(13):  # public to synthetic
        features.append((access_flags >> i) & 1)

    return np.array(features)


def get_instruction_features(node: EncodedMethod) -> np.ndarray:
    """
    Extract features from EncodedMethod node's instructions.

    :param node: EncodedMethod node
    :return: numpy.ndarray[shape=(num_features,)]
    """
    features = []
    instructions = list(node.get_instructions())

    # number of instructions
    features.append(np.array([len(instructions)]))

    # instructions opcodes
    features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([[x.get_name() for x in instructions]])  # type: ignore
        .toarray()[0]  # type: ignore
    )

    # get all strings used in the method
    contains_invalid = False
    strings = []
    for x in instructions:
        if "get_string" in dir(x):
            s = getattr(x, "get_string")()
            s_ = _clean_utf8(s)
            if s != s_:
                contains_invalid = True
            strings.append(s_)

    # check if the method contains invalid UTF-8 strings
    features.append(np.array([contains_invalid]))

    features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([strings])  # type: ignore
        .toarray()[0]  # type: ignore
    )

    # string statistics
    features.append(get_string_statistics(strings))

    return np.concatenate(features)


def _clean_utf8(s: str) -> str:
    """
    Clean the string by removing non-UTF-8 characters.

    :param s: input string
    :return: cleaned string
    """
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def get_string_statistics(strings: List[str]) -> np.ndarray:
    """
    Extract statistics from the list of strings.

    The features are:
    - Number of strings
    - Average string length
    - Histogram of printable characters
    - Entropy of characters in printable strings
    - Number of external paths
        - This is done by counting '/storage/' and '/sdcard/' substrings
        - Internal paths are exclusive to the APK and wouldn't be useful
    - Number of URLs
        - This is done by counting 'http://' and 'https://' substrings
    - Number of IP addresses
        - This is done by counting the number of substrings that match the IP address regex
    - The equivalent of registry in Android
        - '/shared_prefs/', 'Settings.Secure', 'Settings.System', 'Settings.Global' substrings
    - In-memory DEX execution -- detecting potential DEX loading with reflection using 'invoke-*'
        - 'ClassLoader', 'DexFile', 'loadDex', 'loadClass', 'defineClass', 'loadLibrary'

    :param strings: list of strings
    :return: numpy.ndarray[shape=(num_features,)]
    """

    if not strings:
        strings = [""]
    all_strings = "".join(strings)

    # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
    _as = (
        all_strings.replace("\r", "\x80")
        .replace("\n", "\x80")
        .replace("\t", "\x80")
        .replace("\x00", "\x81")
    )
    shifted_string = [ord(b) - 0x20 for b in _as]
    # filter non-ASCII characters (e.g. Chinese characters)
    for i, b in enumerate(shifted_string):
        if b < 0 or b > 95 + 2:
            shifted_string[i] = 95 + 3  # replace with a placeholder
    # +1 for CR LF TAB, +1 for null terminator, +1 for everything else
    c = np.bincount(shifted_string, minlength=96 + 3)  # histogram count
    # distribution of characters in printable strings
    p = c.astype(np.float32) / (c.sum() + 1e-12)
    # entropy of printable strings
    H = np.sum(-p * np.log2(p + 1e-12))

    features = [
        len(strings),  # number of strings
        np.mean([len(x) for x in strings]),  # average string length
        *p,  # histogram of printable characters
        H,  # entropy of printable strings
        len(
            re.findall(r"/storage/|/sdcard/", all_strings, flags=re.IGNORECASE)
        ),  # external paths
        len(re.findall(r"https?://", all_strings, flags=re.IGNORECASE)),  # URLs
        len(
            re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", " ".join(strings))
        ),  # IP addresses
        len(
            re.findall(
                r"/shared_prefs/|Settings\.Secure|Settings\.System|Settings\.Global",
                all_strings,
                flags=re.IGNORECASE,
            )
        ),  # Android "registry"
        len(
            re.findall(
                r"ClassLoader|DexFile|loadDex|loadClass|defineClass|loadLibrary",
                all_strings,
            )
        ),  # in-memory DEX
    ]

    return np.array(features)


def get_method_signature(node) -> Tuple[List[str], str]:
    """
    Extract method signature of a method.

    All basic types are represented by their first letter:
    B - byte
    C - char
    D - double
    F - float
    I - int
    J - long
    L - class
    S - short
    V - void
    Z - boolean
    [ - array

    :param node: method node
    :return param_types: list of parameter types
    :return return_type: return type
    """
    signature = node.get_descriptor()
    param_types = signature[signature.find("(") + 1 : signature.find(")")].split(" ")
    if len(param_types) == 1 and param_types[0] == "":
        param_types = []
    return_type = signature[signature.find(")") + 1 :]
    # sanity check can be done with node.get_information(), but only for non-native methods
    return param_types, return_type


def get_method_signature_features(node) -> np.ndarray:
    """
    Extract features from a method node's signature.

    :param node: method node
    :return: numpy.ndarray[shape=(num_features,)]
    """
    features = []
    parameters, return_type = get_method_signature(node)

    # number of parameters
    features.append(np.array([len(parameters)]))
    # parameter types
    features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([parameters])  # type: ignore
        .toarray()[0]  # type: ignore
    )

    # return type
    features.append(
        FeatureHasher(n_features=50, input_type="string")
        .transform([[return_type] if return_type != "V" else []])  # type: ignore
        .toarray()[0]  # type: ignore
    )

    # number of local registers
    try:
        register_info = node.get_information().get("registers")
    except (AttributeError, KeyError):
        register_info = None

    # register info is only available for non-native internal methods
    if register_info:
        num_local_registers = register_info[1]
        assert (
            num_local_registers == 0 or num_local_registers == node.get_locals()
        ), f"{num_local_registers=} != {node.get_locals()=}"
        features.append(np.array([num_local_registers]))

    return np.concatenate(features)


def get_misc_features(node: EncodedMethod) -> np.ndarray:
    """
    Extract miscellaneous features from EncodedMethod node.
    Check the comments for the meaning of each feature.

    :param node: EncodedMethod node
    :return: numpy.ndarray[shape=(num_features,)]
    """
    features = []
    features.append(node.is_cached_instructions())

    return np.array(features)


def extract_features(node, dx: Analysis) -> np.ndarray:
    """
    Extract features from EncodedMethod node.

    :param node: method node
    :return: numpy.ndarray[shape=(num_features,)]
    """

    # print(node)

    all_features = [get_name_features(node, dx), get_method_signature_features(node)]

    if isinstance(node, EncodedMethod):
        access_flag_features = get_access_flag_features(node)
        is_native = access_flag_features[9]
        is_abstract = access_flag_features[11]

        if not is_native and not is_abstract:
            # cache the local register information
            local_register = all_features[-1][-1:]
            all_features[-1] = all_features[-1][:-1]

        # both non-native and native methods have access flags
        all_features.append(access_flag_features)

        if not is_native and not is_abstract:
            # print(node)
            all_features.extend(
                [
                    local_register,  # type: ignore
                    get_instruction_features(node),
                    get_misc_features(node),
                    get_code_features(node, window, step),
                ]
            )

    # aggregate!
    return np.concatenate(all_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Function Call Graphs from APK files."
    )
    parser.add_argument(
        "--apk_dir",
        type=str,
        help="Directory containing APK files",
        default="",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=100, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the inference server"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global port
    port = args.port  # set the port for the inference server
    
    apk_files = glob(os.path.join(args.apk_dir, "*.apk"))

    Parallel(n_jobs=args.n_jobs)(
        delayed(create_fcg_graph)(apk_path)
        for apk_path in tqdm(apk_files)
    )

if __name__ == "__main__":
    main()
