from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
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
    test_coordinates()
    test_subgroup()
