import json
import os
from argparse import ArgumentParser

import numpy as np
import transformers
from loguru import logger
from tqdm import tqdm

from dataset import ID2CHI, ID2TYPE
from dev_eval import eval
from model.v3_model import BertELModelV3

DATA_DIR = "data"


def softmax(x):
    x *= 10
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _build_dict(mention2id_path, id2entity_path):
    mention2id = dict()
    id2entity = dict()
    with open(os.path.join(DATA_DIR, "kb.json"), "r", encoding="utf-8") as fin:
        kb = [json.loads(line.strip()) for line in fin.readlines()]
        for i, _ in tqdm(enumerate(kb), total=len(kb)):
            id2entity[kb[i]["subject_id"]] = kb[i]
            _mentions = [kb[i]["subject"]]
            for _mention in _mentions:
                if _mention in mention2id.keys():
                    mention2id[_mention].append(kb[i]["subject_id"])
                else:
                    mention2id[_mention] = [kb[i]["subject_id"]]
            for alias_item in kb[i]["alias"]:
                if alias_item in mention2id.keys():
                    mention2id[alias_item].append(kb[i]["subject_id"])
                else:
                    mention2id[alias_item] = [kb[i]["subject_id"]]
    for k, v in mention2id.items():
        mention2id[k] = list(set(v))
    with open(mention2id_path, "w", encoding="utf-8") as fot:
        fot.write(json.dumps(mention2id, ensure_ascii=False))
    with open(id2entity_path, "w", encoding="utf-8") as fot:
        fot.write(json.dumps(id2entity, ensure_ascii=False))


def _load_global_dict():
    """由于这几个文件加载很慢，使用全局加载"""

    _mention2id_path = os.path.join(DATA_DIR, "mention2id.json")
    _id2entity_path = os.path.join(DATA_DIR, "id2entity.json")
    _kb_sum_path = os.path.join(DATA_DIR, "kb_sum.json")

    # _build_dict(_mention2id_path, _id2entity_path)
    mention2id = json.load(open(_mention2id_path, "r", encoding="utf8"))
    logger.info("load mention2id.json, id2entity.json, ......")
    id2entity = json.load(open(_id2entity_path, "r", encoding="utf8"))
    kbsum = json.load(open(_kb_sum_path, "r", encoding="utf8"))
    return mention2id, id2entity, kbsum


mention2id, id2entity, kbsum = _load_global_dict()


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--mode", type=str, default="dev")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--answer", type=str, default="dev_answer.json")
    parser.add_argument(
        "--check_point",
        type=str,
        default="/shao_rui/xyq/nlp_el/demos/check_points/low_lr/best1.ckpt",
    )
    # # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    logger.info(f"Load checkpoint: {args.check_point}")
    logger.info(f"Mode: {args.mode}  Answer Path: {args.answer}")

    with open(f"data/{args.mode}.json", "r", encoding="utf-8") as f:
        test_data = [json.loads(s) for s in f.readlines()[:1000]]
    answer = open(f"data/{args.answer}", "w", encoding="utf-8")

    model = BertELModelV3.load_from_checkpoint(args.check_point)
    # model = BertELModelV2.load_from_checkpoint(args.check_point)
    logger.info("Loaded Model... Start Eval...")
    model.freeze()
    model.eval()

    num = 0

    for test in tqdm(test_data, total=len(test_data)):
        for m in test["mention_data"]:
            mm = m["mention"]
            texta = test["text"].replace(mm, "#" + mm + "#", 1)
            nil_type = model.get_type([texta])
            chi = ID2CHI[nil_type.item()]  # type: ignore
            # texta += " #" + mm + "#的类型是" + chi
            if mm in mention2id:
                kb_ids = mention2id[mm]
                matches = []
                matches2 = []
                for kb_id in kb_ids:
                    textb = kbsum[kb_id]
                    match = model.predict(
                        [texta], [textb], ["#" + mm + "#的类型是" + chi]
                    )
                    matches.append(match.item())
                    # matches2.append(model.predict([texta], [textb]).item())
                # print(kb_ids)
                # print(matches)
                # matches = [max(matches[i], matches[i]) for i in range(len(matches))]
                ma = np.argmax(matches)
                if (
                    matches[ma] >= args.threshold
                ):  # todo how to find the best threshold [0.45 is best]
                    m["kb_id"] = kb_ids[ma]
                else:
                    # nil_type = model.get_type([texta])
                    m["kb_id"] = ID2TYPE[nil_type]
            else:
                # nil_type = model.get_type([texta])
                m["kb_id"] = ID2TYPE[nil_type]
                # print(test, mm)
        answer.write(json.dumps(test, ensure_ascii=False) + "\n")
        if args.mode == "dev":
            answer.flush()
        num += 1
        if num % 1000 == 0:
            if args.mode == "dev":
                print(
                    eval(
                        f"data/{args.mode}.json",
                        f"data/{args.answer}",
                    )
                )
    answer.close()

    if args.mode == "dev":
        print(
            eval(
                f"data/{args.mode}.json",
                f"data/{args.answer}",
            )
        )
