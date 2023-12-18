import json
import os
import random

from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

TYPE2ID = {
    "NIL_Event": 0,
    "NIL_Person": 1,
    "NIL_Work": 2,
    "NIL_Location": 3,
    "NIL_Time&Calendar": 4,
    "NIL_Brand": 5,
    "NIL_Natural&Geography": 6,
    "NIL_Game": 7,
    "NIL_Biological": 8,
    "NIL_Medicine": 9,
    "NIL_Food": 10,
    "NIL_Software": 11,
    "NIL_Vehicle": 12,
    "NIL_Website": 13,
    "NIL_Disease&Symptom": 14,
    "NIL_Organization": 15,
    "NIL_Awards": 16,
    "NIL_Education": 17,
    "NIL_Culture": 18,
    "NIL_Constellation": 19,
    "NIL_Law&Regulation": 20,
    "NIL_VirtualThings": 21,
    "NIL_Diagnosis&Treatment": 22,
    "NIL_Other": 23,
}

ID2CHI = {
    0: "事件活动",
    1: "人物",
    2: "作品",
    3: "区域场所",
    4: "时间历法",
    5: "品牌",
    6: "自然地理",
    7: "游戏",
    8: "生物",
    9: "药物",
    10: "实物",
    11: "软件",
    12: "车辆",
    13: "网站平台",
    14: "疾病症状",
    15: "组织机构",
    16: "奖项",
    17: "教育",
    18: "文化",
    19: "星座",
    20: "法律法规",
    21: "虚拟事物",
    22: "诊断治疗方法",
    23: "其他",
}

ID2TYPE = list(TYPE2ID.keys())

mention2id = None
id2entity = None
kbsum = None

DATA_DIR = "data"


class ELDataset(Dataset):
    def __init__(self, mode) -> None:
        super().__init__()

        self.mode = mode
        self.samples = self._load_data()
        self.mention2id, self.id2entity, self.kb_sum = self._load_global_dict()

        if self.mode in ["train", "dev"]:
            # self.new_stage1()
            # self._add_type_train_dev()
            self.get_stage2()
            self.merge_stage12()
            pass

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return len(self.samples)

    def _load_data(self):
        """load {train}{dev}{test}.json"""
        logger.info(f"loading from {self.mode}.json...")
        with open(os.path.join(DATA_DIR, f"{self.mode}.json"), encoding="utf8") as f:
            samples = [json.loads(line) for line in f.readlines()]
        return samples

    def _build_dict(self, mention2id_path, id2entity_path):
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
        for k, v in mention2id.items():
            mention2id[k] = list(set(v))
        with open(mention2id_path, "w", encoding="utf-8") as fot:
            fot.write(json.dumps(mention2id, ensure_ascii=False))
        with open(id2entity_path, "w", encoding="utf-8") as fot:
            fot.write(json.dumps(id2entity, ensure_ascii=False))

    def _load_global_dict(self):
        """由于这几个文件加载很慢，使用全局加载"""
        global mention2id
        global id2entity
        global kbsum

        _mention2id_path = os.path.join(DATA_DIR, "mention2id.json")
        _id2entity_path = os.path.join(DATA_DIR, "id2entity.json")
        _kb_sum_path = os.path.join(DATA_DIR, "kb_sum.json")

        if mention2id is None:
            # if not os.path.exists(_mention2id_path):
            #     self._build_dict(_mention2id_path, _id2entity_path)
            mention2id = json.load(open(_mention2id_path, encoding="utf8"))
            logger.info("load mention2id.json, id2entity.json, ......")
        if id2entity is None:
            id2entity = json.load(open(_id2entity_path, encoding="utf8"))
        if kbsum is None:
            kbsum = json.load(open(_kb_sum_path, encoding="utf8"))
        return mention2id, id2entity, kbsum

    def _add_type_train_dev(self):
        """为每一个mention添加type"""
        for s in self.samples:
            for mention in s["mention_data"]:
                kb_id = mention["kb_id"]

                if "_" in kb_id:
                    if "|" in kb_id:
                        kb_id = kb_id.split("|")[0]
                    mention["type"] = TYPE2ID[kb_id]
                else:
                    if kb_id not in self.id2entity.keys():
                        mention["type"] = TYPE2ID["NIL_Other"]
                    else:
                        type = self.id2entity[kb_id]["type"]
                        if "|" in type:
                            type = type.split("|")[0]
                        if isinstance(type, str):
                            mention["type"] = TYPE2ID["NIL_" + type]
                        elif isinstance(type, list):
                            mention["type"] = TYPE2ID["NIL_" + type[0]]
        with open(f"data/stage1_{self.mode}.json", "w", encoding="utf-8") as f:
            for s in self.samples:
                text = s["text"]
                for m in s["mention_data"]:
                    f.write(
                        json.dumps(
                            {
                                "text": text + "。" + m["mention"] + ":",
                                "target": m["type"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def get_stage2(self):
        with open("data/kb_sum.json", "r", encoding="utf8") as f:
            id_summarize = json.load(f)
        stage2 = open(f"data/stage2_{self.mode}.json", "w", encoding="utf8")
        for s in self.samples:
            texta = s["text"]
            for m in s["mention_data"]:
                mention = m["mention"]
                if mention not in self.mention2id.keys():
                    continue
                mention_ids = self.mention2id[mention]
                random.shuffle(mention_ids)
                neg_count = 0  # 负样本计数，1:2
                has_pos = False
                for mid in mention_ids:
                    if mid == m["kb_id"] and not has_pos:  # positive
                        stage2.write(
                            json.dumps(
                                {
                                    "texta": texta.replace(
                                        mention, "#" + mention + "#"
                                    ),
                                    "textb": id_summarize[mid],
                                    "label": 1,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        has_pos = True
                    else:  # negative
                        if neg_count == 2:
                            continue
                        else:
                            stage2.write(
                                json.dumps(
                                    {
                                        "texta": texta.replace(
                                            mention, "#" + mention + "#"
                                        ),
                                        "textb": id_summarize[mid],
                                        "label": 0,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            neg_count += 1
        stage2.close()
        logger.info(f"stage2_{self.mode}.json saved.")

    def merge_stage12(self):
        logger.info(f"merge stage1_{self.mode}.json and stage2_{self.mode}.json ...")

        with open("data/new_s1.json", "r", encoding="utf8") as f:
            stage1 = [json.loads(line) for line in f.readlines()]
        print(len(stage1))
        with open(f"data/stage2_{self.mode}.json", "r", encoding="utf8") as f:
            stage2 = [json.loads(line) for line in f.readlines()]
        print(len(stage2))
        # last = 0
        for idx, s2 in tqdm(enumerate(stage2), total=len(stage2)):
            findd = False
            for idx, s1 in enumerate(stage1):
                if s1["text"] == s2["texta"]:
                    stage2[idx]["target"] = s1["target"]
                    # last += idx
                    findd = True
                    break
            if not findd:
                print(s2)
        print(stage2[0])

        with open(f"data/final_{self.mode}.json", "w", encoding="utf8") as f:
            for s in stage2:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                f.flush()

        logger.info(f"stage1_{self.mode}.json and stage2_{self.mode}.json merged.")

    def new_stage1(self):
        with open(
            "/shao_rui/xyq/nlp_el/demos/data/stage1_train.json", "r", encoding="utf8"
        ) as f:
            data = [json.loads(s) for s in f.readlines()]
        for s in tqdm(data):
            text: str = s["text"]
            dot = text.rfind("。")
            m = text[dot + 1 : -1]
            text = text[:dot]
            text = text.replace(m, "#" + m + "#")
            s["text"] = text
        with open("data/new_s1.json", "w", encoding="utf8") as fp:
            for s in data:
                fp.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    data = ELDataset("train")
