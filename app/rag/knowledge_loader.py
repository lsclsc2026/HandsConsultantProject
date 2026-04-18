from pathlib import Path

from app.core.config import ROOT_DIR


class KnowledgeLoader:
    def __init__(self) -> None:
        self.knowledge_dir = ROOT_DIR / "knowledge"
        self.extra_legacy_files = [
            ROOT_DIR / "手相学",
            ROOT_DIR / "手相学.txt",
        ]

    def load_rules(self) -> list[str]:
        docs: list[str] = []

        # 1) 默认规则文件
        rules_file = self.knowledge_dir / "palm_rules.txt"
        if rules_file.exists():
            docs.extend(self._split_paragraphs(rules_file.read_text(encoding="utf-8")))

        # 2) 知识目录中额外规则文件（排除 QA 文件）
        if self.knowledge_dir.exists():
            for file_path in sorted(self.knowledge_dir.glob("*.txt")):
                if file_path.name in {"palm_rules.txt", "qa_cases.txt"}:
                    continue
                docs.extend(self._split_paragraphs(file_path.read_text(encoding="utf-8")))

        # 3) 兼容历史根目录文件
        for legacy_file in self.extra_legacy_files:
            if legacy_file.exists():
                docs.extend(self._split_paragraphs(legacy_file.read_text(encoding="utf-8")))

        return [d for d in docs if d.strip()]

    def load_qa(self) -> list[str]:
        qa_file = self.knowledge_dir / "qa_cases.txt"
        if not qa_file.exists():
            return []
        content = qa_file.read_text(encoding="utf-8")

        pairs: list[str] = []
        question = ""
        answer = ""
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Q:"):
                if question and answer:
                    pairs.append(f"问题：{question}\n回答：{answer}")
                question = line.replace("Q:", "", 1).strip()
                answer = ""
            elif line.startswith("A:"):
                answer = line.replace("A:", "", 1).strip()
            else:
                if answer:
                    answer += " " + line
        if question and answer:
            pairs.append(f"问题：{question}\n回答：{answer}")
        return pairs

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        blocks = [chunk.strip() for chunk in text.split("\n\n")]
        return [b for b in blocks if len(b) >= 20]
