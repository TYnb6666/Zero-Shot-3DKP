"""Tests for KPNetGenerator.__init_subclass__ type-parameter derivation."""
from __future__ import annotations


def test_init_subclass_derives_io() -> None:
    from zerokey.generators.kpnet import KPNetGenerator
    from zerokey.io.kpnet import KPNetIO

    class DummyIO(KPNetIO):
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class DummyModel:
        pass

    class MyGen(KPNetGenerator[DummyIO, DummyModel]):  # type: ignore[type-var]
        pass

    assert MyGen.KPIO is DummyIO


def test_init_subclass_derives_multimodal() -> None:
    from zerokey.generators.kpnet import KPNetGenerator
    from zerokey.io.kpnet import KPNetIO

    class DummyIO(KPNetIO):
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class DummyModel:
        pass

    class MyGen(KPNetGenerator[DummyIO, DummyModel]):  # type: ignore[type-var]
        pass

    assert MyGen.Multimodal is DummyModel


def test_explicit_classvar_takes_precedence() -> None:
    from zerokey.generators.kpnet import KPNetGenerator
    from zerokey.io.kpnet import KPNetIO

    class DummyIO(KPNetIO):
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class DummyIO2(KPNetIO):
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class DummyModel:
        pass

    class MyGen(KPNetGenerator[DummyIO, DummyModel]):  # type: ignore[type-var]
        KPIO = DummyIO2  # type: ignore[assignment]

    # Explicit KPIO in the class body should win over the type parameter
    assert MyGen.KPIO is DummyIO2
