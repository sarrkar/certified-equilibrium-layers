"""Tests for the dataset role registry."""

from __future__ import annotations

import pytest

from cascade.data import (
    EFFICACY_BACKBONE,
    OFFTARGET_CERTIFICATION,
    SYNTHETIC_COVERAGE,
    dataset_role,
    datasets_with_role,
)
from cascade.data.roles import DATASET_ROLES, ROLES


def test_every_dataset_has_a_known_role() -> None:
    assert {EFFICACY_BACKBONE, OFFTARGET_CERTIFICATION, SYNTHETIC_COVERAGE} == ROLES
    assert all(role in ROLES for role in DATASET_ROLES.values())


def test_role_partitions_cover_the_registry() -> None:
    union = (
        set(datasets_with_role(EFFICACY_BACKBONE))
        | set(datasets_with_role(OFFTARGET_CERTIFICATION))
        | set(datasets_with_role(SYNTHETIC_COVERAGE))
    )
    assert union == set(DATASET_ROLES)


def test_known_role_assignments() -> None:
    assert dataset_role("huesken") == EFFICACY_BACKBONE
    assert dataset_role("gse184929") == OFFTARGET_CERTIFICATION
    assert dataset_role("jackson2003") == OFFTARGET_CERTIFICATION
    assert dataset_role("synthetic_star") == SYNTHETIC_COVERAGE


def test_unknown_role_and_dataset_raise() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        datasets_with_role("not_a_role")
    with pytest.raises(KeyError):
        dataset_role("does_not_exist")
