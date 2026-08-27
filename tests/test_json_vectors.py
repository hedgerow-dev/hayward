"""HW-142: RCE vectors transformers carries in standalone JSON config files.

transformers reads its configuration from `*.json` files in a model repo
(tokenizer_config.json, chat_template.json, config.json), never from the model
binary, and two fields in those files reach code execution on load:

- `chat_template`: a Jinja2 template transformers renders. A code-execution
  construct in code position is SSTI -> RCE (MFV-HF-001), the same threat as
  the GGUF-embedded chat_template MFV-GGUF-003 catches.
- `auto_map` / `trust_remote_code`: the HuggingFace-hub remote-code vector
  (MFV-HF-002). auto_map names custom Python the loader imports and runs; a
  config shipping `trust_remote_code: true` asserts its own code is trusted.

Fixtures are hand-built plain dicts written as JSON, the way transformers
would find them on disk, so this file stands alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from hayward.findings import Category, Severity
from hayward.scanner import ModelFileScanner


def _scan_json(tmp_path: Path, name: str, obj: object) -> list:
    """Write `obj` as JSON under `name` and scan it, the shape transformers
    would load from a repo."""
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return ModelFileScanner().scan_file(p)


# ── MFV-HF-001: chat_template SSTI ──────────────────────────────────


class TestChatTemplateSSTI:
    """A Jinja2 chat_template in a JSON config is flagged only when it carries
    a real code-execution construct in code position, never for ordinary
    variable substitution."""

    def test_globals_reach_in_tokenizer_config_fires(self, tmp_path):
        # The canonical SSTI escape: reach __globals__ off a bound method in
        # code position ({{ ... }}), exactly the introspection MFV-GGUF-003
        # catches, here in tokenizer_config.json.
        obj = {
            "chat_template": "{{ self.__init__.__globals__ }}",
            "bos_token": "<s>",
        }
        findings = _scan_json(tmp_path, "tokenizer_config.json", obj)
        hf001 = [f for f in findings if f.rule_id == "MFV-HF-001"]
        assert hf001, [(f.rule_id, f.message) for f in findings]
        assert hf001[0].severity == Severity.CRITICAL
        assert hf001[0].category == Category.SSTI
        assert hf001[0].cwe_ids == [94, 1336]

    def test_os_popen_in_code_position_fires(self, tmp_path):
        obj = {"chat_template": "{{ x.os.popen('id').read() }}"}
        findings = _scan_json(tmp_path, "chat_template.json", obj)
        assert any(f.rule_id == "MFV-HF-001" for f in findings), \
            [(f.rule_id, f.message) for f in findings]

    def test_template_list_form_fires(self, tmp_path):
        # transformers' multi-template form: a list of {name, template}.
        obj = {
            "chat_template": [
                {"name": "default", "template": "{{ message['role'] }}"},
                {"name": "tool_use", "template": "{{ ''.__class__.__mro__ }}"},
            ]
        }
        findings = _scan_json(tmp_path, "tokenizer_config.json", obj)
        assert any(f.rule_id == "MFV-HF-001" for f in findings), \
            [(f.rule_id, f.message) for f in findings]

    def test_benign_substitution_template_does_not_fire(self, tmp_path):
        # An ordinary chat template: pure variable substitution, the shape
        # every real chat-tuned model ships. It must stay clean.
        obj = {
            "chat_template": (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}"
            ),
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        findings = _scan_json(tmp_path, "tokenizer_config.json", obj)
        assert not any(f.rule_id == "MFV-HF-001" for f in findings), \
            [(f.rule_id, f.message) for f in findings]

    def test_signature_word_inside_string_literal_does_not_fire(self, tmp_path):
        # A signature substring appearing inside a {{ '...' }} literal is data,
        # not code (the measured DeepSeek-V4 "scenarios."/os. collision). Code
        # position blanks literals, so this must not fire.
        obj = {"chat_template": "{{ 'run these os. scenarios.' }}"}
        findings = _scan_json(tmp_path, "tokenizer_config.json", obj)
        assert not any(f.rule_id == "MFV-HF-001" for f in findings), \
            [(f.rule_id, f.message) for f in findings]


# ── MFV-HF-002: auto_map / trust_remote_code ────────────────────────


class TestRemoteCodeVector:
    """auto_map and trust_remote_code are the HF-hub remote-code surface."""

    def test_auto_map_fires(self, tmp_path):
        obj = {
            "model_type": "custom",
            "auto_map": {
                "AutoConfig": "configuration_custom.CustomConfig",
                "AutoModelForCausalLM": "modeling_custom.CustomModel",
            },
        }
        findings = _scan_json(tmp_path, "config.json", obj)
        hf002 = [f for f in findings if f.rule_id == "MFV-HF-002"]
        assert hf002, [(f.rule_id, f.message) for f in findings]
        assert hf002[0].severity == Severity.HIGH
        assert hf002[0].category == Category.INJECTION
        assert hf002[0].cwe_ids == [94]
        # The dotted targets are surfaced so a reviewer sees what would import.
        assert "modeling_custom.CustomModel" in hf002[0].metadata["auto_map_targets"]

    def test_trust_remote_code_true_fires(self, tmp_path):
        obj = {"model_type": "custom", "trust_remote_code": True}
        findings = _scan_json(tmp_path, "config.json", obj)
        hf002 = [f for f in findings if f.rule_id == "MFV-HF-002"]
        assert hf002, [(f.rule_id, f.message) for f in findings]
        assert hf002[0].severity == Severity.HIGH

    def test_trust_remote_code_false_does_not_fire(self, tmp_path):
        # The default. Only an explicit `true` is the assertion worth flagging.
        obj = {"model_type": "custom", "trust_remote_code": False}
        findings = _scan_json(tmp_path, "config.json", obj)
        assert not any(f.rule_id == "MFV-HF-002" for f in findings), \
            [(f.rule_id, f.message) for f in findings]


# ── benign configs stay clean ───────────────────────────────────────


class TestBenignConfigs:
    def test_plain_config_json_yields_nothing(self, tmp_path):
        # A real config.json with none of the three fields.
        obj = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }
        findings = _scan_json(tmp_path, "config.json", obj)
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_non_config_json_yields_nothing(self, tmp_path):
        # An unrelated .json object with none of the config fields.
        findings = _scan_json(tmp_path, "data.json", {"a": 1})
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_json_array_is_not_a_config(self, tmp_path):
        # Only a JSON *object* is a config; a top-level array is skipped.
        findings = _scan_json(tmp_path, "list.json", [1, 2, 3])
        assert findings == [], [(f.rule_id, f.message) for f in findings]


class TestDirectoryDiscovery:
    """A directory scan (the normal way a downloaded model repo is checked)
    must discover and scan .json configs, or the HF vectors are reported clean
    on exactly the input they exist to catch."""

    def test_config_json_in_a_subtree_is_found(self, tmp_path):
        repo = tmp_path / "repo" / "nested"
        repo.mkdir(parents=True)
        (repo / "config.json").write_text(json.dumps(
            {"auto_map": {"AutoModel": "modeling_x.Model"}}))
        (repo / "weights.safetensors").write_bytes(b"not a real model")

        findings = ModelFileScanner().scan_directory(tmp_path)
        assert any(f.rule_id == "MFV-HF-002" for f in findings), (
            [(f.rule_id, f.file_path) for f in findings]
        )

    def test_unrelated_json_in_a_tree_is_not_flagged(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps(
            {"name": "app", "version": "1.0.0", "scripts": {"build": "tsc"}}))
        findings = ModelFileScanner().scan_directory(tmp_path)
        assert not any(f.rule_id.startswith("MFV-HF") for f in findings)


class TestAttrFilterEvasion:
    """The |attr filter reassembles a dunder from a string literal, which the
    literal-blanking pass would otherwise hide."""

    def test_attr_filter_reaching_class_fires(self, tmp_path):
        obj = {"chat_template": "{{ ''|attr('__class__') }}"}
        findings = _scan_json(tmp_path, "tokenizer_config.json", obj)
        assert any(f.rule_id == "MFV-HF-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )
