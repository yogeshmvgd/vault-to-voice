from pydantic import BaseModel

from tests.fakes import FakeLLMClient


class A(BaseModel):
    x: int


def test_static_list_and_callable():
    fake = FakeLLMClient({A: [A(x=1), A(x=2)]})
    assert [fake.structured("p", A).x for _ in range(2)] == [1, 2]
    fake2 = FakeLLMClient({A: lambda prompt: A(x=len(prompt))})
    assert fake2.structured("abc", A).x == 3
    assert fake2.calls == [(A, "abc")]
