import json

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing


BASE_VOCAB = [
    "<pad>", "<unk>", "<mask>", "<bos>", "<eos>", "|",
    "X", "B", "O", "U", "Z", "J", *"LAGVSERTIDPKQNFYMHWC",
]


def feature_columns(dimension):
    return [f"embedding_{index:04d}" for index in range(dimension)]


def write_tokenizer(root, vocab=BASE_VOCAB):
    root.mkdir(exist_ok=True)
    tokens = {token: index for index, token in enumerate(vocab)}
    tokenizer = Tokenizer(WordPiece(vocab=tokens, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split("", behavior="removed")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos>:0 $A:0 <eos>:0",
        special_tokens=[("<bos>", tokens["<bos>"]), ("<eos>", tokens["<eos>"])],
    )
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({
        f"{role}_token": f"<{role}>" for role in ("pad", "mask", "bos", "eos")
    }))
    return root
