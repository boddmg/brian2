from nose.plugins.attrib import attr
from numpy.testing.utils import (assert_equal, assert_allclose, assert_raises,
                                 assert_array_equal)

import numpy as np

from brian2.spatialneuron import *
from brian2.units import um, second

@attr('codegen-independent')
def test_basicshapes():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    morpho.right['nextone'] = Cylinder(length=2*um, diameter=1*um, n=3)
    # Check total number of compartments
    assert_equal(len(morpho),26)
    assert_equal(len(morpho.L.main),10)
    assert_allclose(morpho.LL.distance[-1], 15*um)

@attr('codegen-independent')
def test_modular_construction():
    morpho1 = Cylinder(length=10*um, diameter=1*um, n=1)
    morpho1.L = Cylinder(length=10*um, diameter=1*um, n=1)
    morpho1.LL = Cylinder(length=10*um, diameter=1*um, n=1)

    sub_morpho = Cylinder(length=10*um, diameter=1*um, n=1)
    sub_morpho.L = Cylinder(length=10*um, diameter=1*um, n=1)
    morpho2 = Cylinder(length=10*um, diameter=1*um, n=1)
    morpho2.L = sub_morpho

    # The two ways of constructing a morphology should be equivalent
    assert_equal(len(morpho1), len(morpho2))
    assert_allclose(morpho1.distance, morpho2.distance)
    assert_allclose(morpho1.L.distance, morpho2.L.distance)
    assert_allclose(morpho1.LL.distance, morpho2.LL.distance)
    assert_allclose(morpho1.length, morpho2.length)
    assert_allclose(morpho1.L.length, morpho2.L.length)
    assert_allclose(morpho1.LL.length, morpho2.LL.length)
    assert_allclose(morpho1.area, morpho2.area)
    assert_allclose(morpho1.L.area, morpho2.L.area)
    assert_allclose(morpho1.LL.area, morpho2.LL.area)

@attr('codegen-independent')
def test_from_segments():
    # A simple structure:
    #      -- 1        -- 4
    # 0 --/-- 2 -- 3 -/
    #                 \
    #                  -- 5 -- 6
    segments = [
        dict(T='soma', x=0*um, y=0*um, z=0*um, diameter=30*um, children=[1, 2]),    # 0
        dict(T='axon', x=10*um, y=0*um, z=0*um, diameter=1*um, children=[]),        # 1
        dict(T='dendrite', x=0*um, y=10*um, z=0*um, diameter=5*um, children=[3]),   # 2
        dict(T='dendrite', x=0*um, y=15*um, z=0*um, diameter=5*um, children=[4, 5]),# 3
        dict(T='dendrite', x=10*um, y=15*um, z=0*um, diameter=2*um, children=[]),   # 4
        dict(T='dendrite', x=0*um, y=15*um, z=5*um, diameter=3*um, children=[6]),   # 5
        dict(T='dendrite', x=0*um, y=15*um, z=15*um, diameter=3*um, children=[])    # 6
    ]
    morpho = Morphology.from_segments(segments)

    assert len(morpho) == 7
    # Soma
    assert_array_equal(morpho.length, [0]*um)
    assert_array_equal(morpho.diameter, [30]*um)
    assert_allclose(morpho.area, ([30]*um)**2*np.pi)
    assert_array_equal(morpho.x, [0]*um)
    assert_array_equal(morpho.y, [0]*um)
    assert_array_equal(morpho.z, [0]*um)
    assert_array_equal(morpho.distance, [0]*um)
    assert isinstance(morpho, Soma)

    # Axon
    assert_array_equal(morpho.axon.length, [10]*um)
    assert_array_equal(morpho.axon.diameter, [1]*um)
    assert_allclose(morpho.axon.area,
                    morpho.axon.length * morpho.axon.diameter * np.pi)
    assert_array_equal(morpho.axon.x, [10]*um)
    assert_array_equal(morpho.axon.y, [0]*um)
    assert_array_equal(morpho.axon.z, [0]*um)
    assert_array_equal(morpho.axon.distance, [10]*um)

    # First dendrite (branching from soma)
    assert_array_equal(morpho.dendrite.length, [10, 5]*um)
    assert_array_equal(morpho.dendrite.diameter, [5, 5]*um)
    assert_allclose(morpho.dendrite.area,
                    morpho.dendrite.length * morpho.dendrite.diameter * np.pi)
    assert_array_equal(morpho.dendrite.x, [0, 0]*um)
    assert_array_equal(morpho.dendrite.y, [10, 15]*um)
    assert_array_equal(morpho.dendrite.z, [0, 0]*um)
    assert_array_equal(morpho.dendrite.distance, [10, 15]*um)
    assert isinstance(morpho.dendrite, Cylinder)

    # Upper dendrite (branching from first dendrite)
    assert morpho.dendrite.L == morpho.dendrite['1']
    assert_array_equal(morpho.dendrite.L.length, [10]*um)
    assert_array_equal(morpho.dendrite.L.diameter, [2]*um)
    assert_allclose(morpho.dendrite.L.area,
                    morpho.dendrite.L.length * morpho.dendrite.L.diameter * np.pi)
    assert_array_equal(morpho.dendrite.L.x, [10]*um)
    assert_array_equal(morpho.dendrite.L.y, [15]*um)
    assert_array_equal(morpho.dendrite.L.z, [0]*um)
    assert_array_equal(morpho.dendrite.L.distance, [25]*um)
    assert isinstance(morpho.dendrite.L, Cylinder)
    # There should be no name "dendrite" here as it is ambiguous
    assert "dendrite" not in morpho.dendrite._named_children

    # Lower dendrite (branching from first dendrite)
    assert morpho.dendrite.R == morpho.dendrite['2']
    assert_array_equal(morpho.dendrite.R.length, [5, 10]*um)
    assert_array_equal(morpho.dendrite.R.diameter, [3, 3]*um)
    assert_allclose(morpho.dendrite.R.area,
                    morpho.dendrite.R.length * morpho.dendrite.R.diameter * np.pi)
    assert_array_equal(morpho.dendrite.R.x, [0, 0]*um)
    assert_array_equal(morpho.dendrite.R.y, [15, 15]*um)
    assert_array_equal(morpho.dendrite.R.z, [5, 15]*um)
    assert_array_equal(morpho.dendrite.R.distance, [20, 30]*um)
    assert isinstance(morpho.dendrite.R, Cylinder)


def _check_consistency(morphology):
    '''
    Helper function to check that area, distance and coordinates are consistent
    with length and diameter (assuming a cylinder)
    '''
    assert isinstance(morphology, Cylinder)
    assert_allclose(morphology.area, np.pi*morphology.length*morphology.diameter)
    if morphology._parent is None:
        parent_dist = 0*um
        parent_x = parent_y = parent_z = 0*um
    else:
        parent_dist = morphology._parent.distance[-1]
        parent_x = morphology._parent.x[-1]
        parent_y = morphology._parent.y[-1]
        parent_z = morphology._parent.z[-1]
    assert_allclose(morphology.distance, parent_dist + np.cumsum(morphology.length))

    diff_loc = np.diff(np.hstack([np.asarray([[parent_x], [parent_y], [parent_z]]),
                       np.asarray([morphology.x, morphology.y, morphology.z])]))
    assert_allclose(np.sqrt(np.sum(diff_loc**2, axis=0)),
                    np.asarray(morphology.length))

@attr('codegen-independent')
def test_change_n():
    soma = Soma(diameter=30*um)
    soma.axon = Cylinder(length=100*um, diameter=10*um, n=10)
    soma.dendrite = Cylinder(length=100*um, diameter=1*um, n=10)
    soma.dendrite.L = Cylinder(length=100*um, diameter=1*um, n=10)
    soma.dendrite.R = Cylinder(length=100*um, diameter=1*um, n=10)

    # Changing the number of compartments of a soma should not work
    assert_raises(AttributeError, lambda: setattr(soma, 'n', 2))

    attributes = ['length', 'diameter', 'area', 'distance', 'x', 'y', 'z']

    soma.dendrite.n = 20  # Split compartments
    assert soma.dendrite.n == 20
    assert all(len(getattr(soma.dendrite, attr)) == 20
               for attr in attributes), [(attr, len(getattr(soma.dendrite, attr))) for attr in attributes]
    assert_allclose(soma.dendrite.length, 5*um)
    assert_allclose(soma.dendrite.diameter, 1*um)
    _check_consistency(soma.dendrite)

    soma.dendrite.n = 5  # Merge compartments
    assert soma.dendrite.n == 5
    assert all(len(getattr(soma.dendrite, attr)) == 5
               for attr in attributes), [(attr, len(getattr(soma.dendrite, attr))) for attr in attributes]
    assert_allclose(soma.dendrite.length, 20*um)
    assert_allclose(soma.dendrite.diameter, 1*um)
    _check_consistency(soma.dendrite)


@attr('codegen-independent')
def test_coordinates():
    # All of those should be identical when looking at length and area (all
    # that matters for simulation)
    morpho1 = Cylinder(x=10*um, y=0*um, z=0*um, diameter=1*um, n=10)
    morpho2 = Cylinder(x=0*um, y=10*um, z=0*um, diameter=1*um, n=10)
    morpho3 = Cylinder(x=0*um, y=0*um, z=10*um, diameter=1*um, n=10)
    morpho4 = Cylinder(length=10*um, diameter=1*um, n=10)
    assert_allclose(morpho1.length, morpho2.length)
    assert_allclose(morpho2.length, morpho3.length)
    assert_allclose(morpho3.length, morpho4.length)
    assert_allclose(morpho1.area, morpho2.area)
    assert_allclose(morpho2.area, morpho3.area)
    assert_allclose(morpho3.area, morpho4.area)

    # Check that putting morphologies together correctly updates the coordinates
    morpho2.L = morpho3
    morpho1.L = morpho2
    assert_allclose([morpho3.x[-1], morpho3.y[-1], morpho3.z[-1]],
                    [10*um, 10*um, 10*um])

@attr('codegen-independent')
def test_subgroup():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    # Getting a single compartment by index
    assert_allclose(morpho.L[2].distance,3*um)
    # Getting a single compartment by position
    assert_allclose(morpho.LL[0*um].distance,11*um)
    assert_allclose(morpho.LL[1*um].distance,11*um)
    assert_allclose(morpho.LL[1.5*um].distance,12*um)
    assert_allclose(morpho.LL[5*um].distance,15*um)
    # Getting a segment
    assert_allclose(morpho.L[3*um:5.1*um].distance,[3*um,4*um,5*um])
    # Absolute indices
    assert_equal(morpho.LL.indices[:], [11, 12, 13, 14, 15])
    assert_equal(morpho.L.indices[3*um:5.1*um], [3, 4, 5])
    assert_equal(morpho.L.indices[3*um:5.1*um],
                 morpho.L[3*um:5.1*um].indices[:])
    assert_equal(morpho.L.indices[:5.1*um], [1, 2, 3, 4, 5])
    assert_equal(morpho.L.indices[3*um:], [3, 4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[3.5*um], 4)
    assert_equal(morpho.L.indices[3], 4)
    assert_equal(morpho.L.indices[-1], 10)
    assert_equal(morpho.L.indices[3:5], [4, 5])
    assert_equal(morpho.L.indices[3:], [4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[:5], [1, 2, 3, 4, 5])

    # Main branch
    assert_equal(len(morpho.L.main), 10)

    # Non-existing branch
    assert_raises(AttributeError, lambda: morpho.axon)

    # Incorrect indexing
    #  wrong units or mixing units
    assert_raises(TypeError, lambda: morpho.indices[3*second:5*second])
    assert_raises(TypeError, lambda: morpho.indices[3.4:5.3])
    assert_raises(TypeError, lambda: morpho.indices[3:5*um])
    assert_raises(TypeError, lambda: morpho.indices[3*um:5])
    #   providing a step
    assert_raises(TypeError, lambda: morpho.indices[3*um:5*um:2*um])
    assert_raises(TypeError, lambda: morpho.indices[3:5:2])
    #   incorrect type
    assert_raises(TypeError, lambda: morpho.indices[object()])


if __name__ == '__main__':
    test_basicshapes()
    test_modular_construction()
    test_from_segments()
    test_change_n()
    test_coordinates()
    test_subgroup()
