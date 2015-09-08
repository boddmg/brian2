'''
Neuronal morphology module.
This module defines classes to load and build neuronal morphologies.
'''
from copy import copy as stdlib_copy
import numbers

from numpy.random import rand

from brian2.numpy_ import *
from brian2.units.allunits import meter
from brian2.utils.logger import get_logger
from brian2.units.stdunits import um
from brian2.units.fundamentalunits import (have_same_dimensions, Quantity,
                                           check_units)

logger = get_logger(__name__)

__all__ = ['Morphology', 'Cylinder', 'Soma']


def hash_array(arr):
    return hash(memoryview(arr).tobytes())


def _set_root(morphology, root):
    # Recursively set the `_root` attribute to link to the main morphology
    # object
    morphology._root = root
    for child in morphology.children:
        _set_root(child, root)


class MorphologyIndexWrapper(object):
    '''
    A simpler version of `~brian2.groups.group.IndexWrapper`, not allowing for
    string indexing (`Morphology` is not a `Group`). It allows to use
    ``morphology.indices[...]`` instead of ``morphology[...]._indices()``.
    '''
    def __init__(self, morphology):
        self.morphology = morphology

    def __getitem__(self, item):
        if isinstance(item, basestring):
            raise NotImplementedError(('Morphologies do not support string '
                                       'indexing'))
        return self.morphology._indices(item)


class Morphology(object):
    '''
    Neuronal morphology (=tree of branches).

    The data structure is a tree where each node is a segment consisting
    of a number of connected compartments, each one defined by its geometrical properties
    (length, area, diameter, position).

    Parameters
    ----------
    filename : str, optional
        The name of a swc file defining the morphology.
        If not specified, makes a segment (if `n` is specified) or an empty morphology.
    n : int, optional
        Number of compartments.
    '''
    # Specifing slots makes it easier to figure out which assignments are meant
    # to create a subtree (see `__setattr__`)
    __slots__ = ['children', '_named_children', 'indices', 'type',
                 '_origin', '_root', '_n', '_x', '_y', '_z', '_diameter',
                 '_length', '_area', '_distance', '_parent']

    def __init__(self, filename=None, n=None, parent=None):
        self.children = []
        self._named_children = {}
        self.indices = MorphologyIndexWrapper(self)
        self.type = None
        self._origin = 0
        self._root = self
        self._parent = parent
        if filename is not None:
            if n is not None:
                raise TypeError(('Cannot specify the number of compartments '
                                 'when loading a morphology from a file.'))
            self.loadswc(filename)
        else:
            if n is None:
                n = 0  # Create an empty morphology that will be filled later
            (self._x, self._y, self._z, self._diameter, self._length,
             self._area, self._distance) = [zeros(n) * meter for _ in range(7)]
            self._n = n

    def _update_area(self):
        self._area = pi * self.diameter * self.length

    # All attributes depend on each other, therefore only allow to change them
    # using properties
    # TODO: Check correct shape/type of the arguments
    def _set_n(self, n):
        if n == self.n:
            return  # nothing to do

        if self._parent is None:
            prev_x = prev_y = prev_z =  0.
        else:
            prev_x = float(self._parent.x[-1])
            prev_y = float(self._parent.y[-1])
            prev_z = float(self._parent.z[-1])

        # Only allow splitting up compartments/merging neighbouring compartments
        if n > self.n:
            if n % self.n != 0:
                raise NotImplementedError(('New number of compartments was %d, '
                                           'but has to be a multiple of the '
                                           'old number of compartments '
                                           '%d.') % (n, self.n))
            # Split up each compartment into identical compartments
            split_into = n / self.n
            self._length = self._length.repeat(split_into) / split_into
            self._diameter = self._diameter.repeat(split_into)
            diff_x = diff(hstack([prev_x, asarray(self._x)]))
            diff_y = diff(hstack([prev_y, asarray(self.y)]))
            diff_z = diff(hstack([prev_z, asarray(self.z)]))
            self._x = prev_x + cumsum(diff_x.repeat(split_into) / split_into)
            self._y = prev_y + cumsum(diff_y.repeat(split_into) / split_into)
            self._z = prev_z + cumsum(diff_z.repeat(split_into) / split_into)
        else:
            if self.n % n != 0:
                raise NotImplementedError(('The old number of compartments was '
                                           '%d, which is not a multiple of the '
                                           'given number of compartments '
                                           '%d') % (self.n, n))
            merge_together = self.n / n
            self._length = sum(self.length.reshape(-1, merge_together), axis=1)
            self._diameter = mean(self.diameter.reshape(-1, merge_together), axis=1)
            # Use the average direction of the previous point and use it with
            # the new length
            dir_x = diff(hstack([prev_x, asarray(self.x)])).reshape(-1, merge_together).mean(axis=1)
            dir_y = diff(hstack([prev_y, asarray(self.y)])).reshape(-1, merge_together).mean(axis=1)
            dir_z = diff(hstack([prev_z, asarray(self.z)])).reshape(-1, merge_together).mean(axis=1)
            # normalize length to 1
            old_length = sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            old_end_x, old_end_y, old_end_z = self.x[-1], self.y[-1], self.z[-1]
            self._x = prev_x + cumsum(dir_x / old_length * self.length)
            self._y = prev_y + cumsum(dir_y / old_length * self.length)
            self._z = prev_z + cumsum(dir_z / old_length * self.length)
            # Update the change from the previous end point of the branch

            for child in self.children:
                # Note that the distance should not change, we kept the total
                # length the same
                child._update_distances_and_coordinates(distance=0,
                                                        x=self.x[-1]-old_end_x,
                                                        y=self.y[-1]-old_end_y,
                                                        z=self.z[-1]-old_end_z)
        self._n = n
        self._update_area()

    n = property(fget=lambda self: self._n,
                 fset=_set_n,
                 doc='The number of compartments in this section')

    def _set_length(self, length):
        if self._parent is None:
            prev_x = prev_y = prev_z = distance = 0.
        else:
            prev_x = float(self._parent.x[-1])
            prev_y = float(self._parent.y[-1])
            prev_z = float(self._parent.z[-1])
            distance = float(self._parent.distance[-1])

        old_end_x, old_end_y, old_end_z = self._x[-1], self._y[-1], self._z[-1]
        diff_x = diff(hstack([prev_x, asarray(self.x)]))
        diff_y = diff(hstack([prev_y, asarray(self.y)]))
        diff_z = diff(hstack([prev_z, asarray(self.z)]))
        # Random direction for new coordinates (where length was zero previously)
        theta = rand() * 2 * pi
        phi = rand() * 2 * pi
        diff_x[self._length == 0*um] = sin(theta) * cos(phi)
        diff_y[self._length == 0*um] = sin(theta) * sin(phi)
        diff_z[self._length == 0*um] = cos(theta)
        scale_by = Quantity(length, copy=True)
        scale_by[self._length > 0*um] = length[self._length > 0*um] / self._length[self._length > 0*um]
        diff_x *= scale_by
        diff_y *= scale_by
        diff_z *= scale_by
        self._x = Quantity(prev_x + diff_x, dim=meter.dim)
        self._y = Quantity(prev_y + diff_y, dim=meter.dim)
        self._z = Quantity(prev_z + diff_z, dim=meter.dim)
        self._distance = cumsum(length) + distance*meter
        # in the update, we only have to propagate the change, not the absolute
        # value
        length_change = cumsum(length)[-1] - cumsum(self._length)[-1]
        x_change = self._x[-1] = old_end_x
        y_change = self._y[-1] = old_end_y
        z_change = self._z[-1] = old_end_z
        # Propagate the changes to the children
        for child in self.children:
            child._update_distances_and_coordinates(distance=length_change,
                                                    x=x_change,
                                                    y=y_change,
                                                    z=z_change)

    length = property(fget=lambda self: self._length,
                      fset=_set_length,
                      doc='The length of each compartment in this section')

    def _set_diameter(self, diameter):
        self._diameter = diameter
        self._update_area()

    diameter = property(fget=lambda self: self._diameter,
                        fset=_set_diameter,
                        doc='The diameter of each compartment in this section')

    # Distance and area are calculated, they cannot be set directly
    area = property(fget=lambda self: self._area,
                    doc='The surface area of each compartment in this section')

    distance = property(fget=lambda self: self._distance,
                        doc='The distance to the root of the morphology for '
                            'each section')

    x = property(fget=lambda self: self._x,
                 doc='The x-coordinate of the end of each compartment in this '
                     'section (relative to the root of the morphology)')
    y = property(fget=lambda self: self._y,
                 doc='The x-coordinate of the end of each compartment in this '
                     'section (relative to the root of the morphology)')
    z = property(fget=lambda self: self._z,
                 doc='The x-coordinate of the end of each compartment in this '
                     'section (relative to the root of the morphology)')
    # TODO: Allow to set the coordinates

    def __hash__(self):
        hash_value = (self.n +
                      hash_array(self.diameter) +
                      hash_array(self.length) +
                      hash_array(self.area) +
                      hash_array(self.distance) +
                      hash(self.type))
        hash_value += hash(frozenset(self._named_children.keys()))
        for child in self.children:
            hash_value += hash(child)
        return hash_value

    def _update_distances_and_coordinates(self, distance=0*um,
                                          x=0*um, y=0*um, z=0*um):
        self._distance += distance
        self._x += x
        self._y += y
        self._z += z
        for child in self.children:
            child._update_distances_and_coordinates(distance, x, y, z)

    def set_coordinates(self):
        '''
        Sets the coordinates of compartments according to their lengths (taking
        a random direction)
        '''
        l = cumsum(self.length)
        theta = rand() * 2 * pi
        phi = rand() * 2 * pi
        self.x = l * sin(theta) * cos(phi)
        self.y = l * sin(theta) * sin(phi)
        self.z = l * cos(theta)

    def loadswc(self, filename):
        '''
        Reads a SWC file containing a neuronal morphology.
        Large database at http://neuromorpho.org/neuroMorpho        
        Information below from http://www.mssm.edu/cnic/swc.html
        
        SWC File Format
        
        The format of an SWC file is fairly simple. It is a text file consisting of a header with various fields beginning with a # character, and a series of three dimensional points containing an index, radius, type, and connectivity information. The lines in the text file representing points have the following layout. 
        n T x y z R P
        n is an integer label that identifies the current point and increments by one from one line to the next.
        T is an integer representing the type of neuronal segment, such as soma, axon, apical dendrite, etc. The standard accepted integer values are given below:
        * 0 = undefined
        * 1 = soma
        * 2 = axon
        * 3 = dendrite
        * 4 = apical dendrite
        * 5 = fork point
        * 6 = end point
        * 7 = custom

        x, y, z gives the cartesian coordinates of each node.
        R is the radius at that node.
        P indicates the parent (the integer label) of the current point or -1 to indicate an origin (soma).

        By default, the soma is assumed to have spherical geometry. If several compartments
        '''
        # 1) Create the list of segments, each segment has a list of children
        lines = open(filename).read().splitlines()
        segment = []  # list of segments
        types = ['undefined', 'soma', 'axon', 'dendrite', 'apical', 'fork',
                 'end', 'custom']
        previousn = -1

        # First pass: determine whether the soma should have spherical or cylindrical geometry.
        # Criterion: spherical if only 1 somatic compartment, otherwise cylindrical
        spherical_soma = len([line.split()[1] == 1 for line in lines if line[0] != '#'])==1

        # Second pass: construction
        for line in lines:
            if line[0] != '#':  # comment
                numbers = line.split()
                n = int(numbers[0]) - 1
                T = types[int(numbers[1])]
                x = float(numbers[2]) * um
                y = float(numbers[3]) * um
                z = float(numbers[4]) * um
                R = float(numbers[5]) * um
                P = int(numbers[6]) - 1  # 0-based indexing
                if (n != previousn + 1):
                    raise ValueError, "Bad format in file " + filename
                seg = dict(x=x, y=y, z=z, T=T, diameter=2 * R, parent=P,
                           children=[])
                location = (x, y, z)
                if (T == 'soma') and spherical_soma: # Assuming spherical geometry
                    seg['area'] = 4 * pi * R ** 2
                    seg['length'] = 0 * um
                elif P==-2: # No parent: we make a zero-length cylinder (not very elegant)
                    seg['area'] = 0 * um**2
                    seg['length'] = 0 * um
                else:  # not soma, dendrite
                    try:
                        locationP = (segment[P]['x'], segment[P]['y'], segment[P]['z'])
                    except IndexError:
                        raise IndexError("When adding segment "+str(n)+": Segment "+str(P)+" does not exist")
                    seg['length'] = (sum((array(location) -
                                          array(locationP)) ** 2)) ** .5 * meter
                    seg['area'] = seg['length'] * 2 * pi * R
                if P >= 0:
                    segment[P]['children'].append(n)
                segment.append(seg)
                previousn = n
        # We assume that the first segment is the root
        self._do_create_from_segments(segment)
        self._update_indices()
        self._update_distances_and_coordinates()

    @classmethod
    def create_from_segments(cls, segments):
        obj = cls()
        obj._do_create_from_segments(segments)
        return obj

    def _do_create_from_segments(self, segments, origin=0):
        """
        Recursively create the morphology from a list of segments.
        Each segments has attributes: x,y,z,diameter,area,length (vectors)
        and children (list).
        It also creates a dictionary of names (_named_children).
        """
        n = origin
        if segments[origin]['T'] != 'soma':  # if it's a soma, only one compartment
            while (len(segments[n]['children']) == 1) and (
                segments[n]['T'] != 'soma'):  # Go to the end of the branch
                n += 1
        # End of branch
        branch = segments[origin:n + 1]
        # Set attributes
        self._n = len(branch)
        self._diameter, self._length, self._area, self._x, self._y, self._z = \
            zip(*[(seg['diameter'], seg['length'], seg['area'], seg['x'],
                   seg['y'], seg['z']) for seg in branch])
        self.type = segments[n]['T']  # normally same type for all compartments
                                     # in the branch
        self._distance = cumsum(self.length)
        # Create children (list)
        self.children = [Morphology(parent=self)._do_create_from_segments(segments, origin=c)
                         for c in segments[n]['children']]
        # Create dictionary of names (enumerates children from number 1)
        for i, child in enumerate(self.children):
            self._named_children[str(i + 1)] = child
            # Name the child if possible
            if child.type in ['soma', 'axon', 'dendrite']:
                if child.type in self._named_children:
                    self._named_children[child.type] = None  # two children with the
                                                       # same name: erase
                                                       # (see next block)
                else:
                    self._named_children[child.type] = child
        # Erase useless names
        for k in self._named_children.keys():
            if self._named_children[k] is None:
                del self._named_children[k]
        # If two children, name them L (left) and R (right)
        if len(self.children) == 2:
            self._named_children['L'] = self._named_children['1']
            self._named_children['R'] = self._named_children['2']

        return self

    def _branch(self):
        '''
        Returns the current branch without the children.
        '''
        morpho = stdlib_copy(self)
        morpho.children = []
        morpho._named_children = {}
        morpho.indices = MorphologyIndexWrapper(morpho)
        return morpho

    def _indices(self, item=None, index_var='_idx'):
        '''
        Returns compartment indices for the main branch, relative to the
        original morphology.
        '''
        if index_var != '_idx':
            raise AssertionError('Unexpected index %s' % index_var)
        if not (item is None or item == slice(None)):
            return self[item]._indices()
        elif self.n == 1:
            return self._origin  # single compartment
        else:
            return arange(self._origin, self._origin + self.n)

    def __getitem__(self, item):
        """
        Returns the subtree named x.
        Ex.: ```neuron['axon']``` or ```neuron['11213']```
        ```neuron[10*um:20*um]``` returns the subbranch from 10 um to 20 um.
        ```neuron[10*um]``` returns one compartment.
        ```neuron[5]``` returns compartment number 5.
        """
        if isinstance(item, slice):  # neuron[10*um:20*um] or neuron[1:3]
            using_lengths = all([arg is None or have_same_dimensions(arg, meter)
                                 for arg in [item.start, item.stop]])
            using_ints = all([arg is None or int(arg) == float(arg)
                                 for arg in [item.start, item.stop]])
            if not (using_lengths or using_ints):
                raise TypeError('Index slice has to use lengths or integers')

            morpho = self._branch()
            if using_lengths:
                if item.step is not None:
                    raise TypeError(('Cannot provide a step argument when '
                                     'slicing with lengths'))
                l = cumsum(array(morpho.length))  # coordinate on the branch
                if item.start is None:
                    i = 0
                else:
                    i = searchsorted(l, float(item.start))
                if item.stop is None:
                    j = len(l)
                else:
                    j = searchsorted(l, float(item.stop))
            else:  # integers
                i, j, step = item.indices(len(morpho))
                if step != 1:
                    raise TypeError('Can only slice a contiguous segment')
        elif isinstance(item, Quantity) and have_same_dimensions(item, meter):  # neuron[10*um]
            morpho = self._branch()
            l = cumsum(array(morpho.length))
            i = searchsorted(l, item)
            j = i + 1
        elif isinstance(item, numbers.Integral):  # int: returns one compartment
            morpho = self._branch()
            if item < 0:  # allows e.g. to use -1 to get the last compartment
                item += len(morpho)
            if item >= len(morpho):
                raise IndexError(('Invalid index %d '
                                  'for %d compartments') % (item, len(morpho)))
            i = item
            j = i + 1
        elif item == 'main':
            return self._branch()
        elif isinstance(item, basestring):
            item = str(item)  # convert int to string
            if (len(item) > 1) and all([c in 'LR123456789' for c in
                                     item]):  # binary string of the form LLLRLR or 1213 (or mixed)
                return self._named_children[item[0]][item[1:]]
            elif item in self._named_children:
                return self._named_children[item]
            else:
                raise AttributeError, "The subtree " + item + " does not exist"
        else:
            raise TypeError('Index of type %s not understood' % type(item))

        # Return the sub-morphology
        morpho._diameter = morpho.diameter[i:j]
        morpho._length = morpho.length[i:j]
        morpho._area = morpho.area[i:j]
        morpho._x = morpho.x[i:j]
        morpho._y = morpho.y[i:j]
        morpho._z = morpho.z[i:j]
        morpho._distance = morpho.distance[i:j]
        morpho._n = j-i
        morpho._origin += i
        return morpho

    def _add_new_child(self, child):
        child._parent = self
        self.children.append(child)
        self._named_children[str(len(self.children))] = child  # numbered child
        _set_root(child, self._root)
        # go up to the parent and update the absolute indices
        self._root._update_indices()
        # Update the distances and coordinates in the subtree -- all previous
        # values are considered being relative to this root
        child._update_distances_and_coordinates(distance=self.distance[-1],
                                                x=self.x[-1],
                                                y=self.y[-1],
                                                z=self.z[-1])

    def __setitem__(self, key, value):
        """
        Inserts the subtree and name it ``item``.
        Ex.: ``neuron['axon']`` or ``neuron['11213']``
        If the tree already exists with another name, then it creates a synonym
        for this tree.
        The coordinates of the subtree are relative before function call,
        and are absolute after function call.
        """
        key = str(key)  # convert int to string
        if key in self._named_children:
            raise AttributeError, "The subtree " + key + " already exists"
        elif key == 'main':
            raise AttributeError, "The main branch cannot be changed"
        elif (len(key) > 1) and all([c in 'LR123456789' for c in key]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            self._named_children[key[0]][key[1:]] = value
        else:
            if value not in self.children:
                self._add_new_child(value)
            self._named_children[key] = value

    def __delitem__(self, item):
        """
        Removes the subtree `item`.
        """
        item = str(item)  # convert int to string
        if (len(item) > 1) and all([c in 'LR123456789' for c in item]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            del self._named_children[item[0]][item[1:]]
        elif item in self._named_children:
            delete_child = self._named_children[item]
            # Delete from name dictionary
            for name, child in self._named_children.iteritems():
                if child is delete_child: del self._named_children[name]
            # Delete from list of children
            for i, child in enumerate(self.children):
                if child is delete_child:
                    del self.children[i]
                    break
        else:
            raise AttributeError('The subtree ' + item + ' does not exist')

        # go up to the parent and update the absolute indices
        self._root._update_indices_and_distances()

    def __getattr__(self, item):
        """
        Returns the subtree named `x`.
        Ex.: ``axon=neuron.axon``
        """
        if item in self.__class__.__slots__ or item in ('diameter', 'length', 'area',
                                                  'distance', 'x', 'y', 'z', 'n'):
            return object.__getattribute__(self, item)
        else:
            return self[item]

    def __setattr__(self, key, value):
        """
        Attach a subtree and name it ``key``. If the subtree is ``None`` then the
        subtree ``key`` is deleted.
        Ex.: ``neuron.axon = Soma(diameter=10*um)``
        Ex.: ``neuron.axon = None``
        """
        if key in self.__slots__ or key in ('diameter', 'length', 'area',
                                        'distance', 'x', 'y', 'z', 'n'):
            object.__setattr__(self, key, value)
        elif value is None:
            del self[key]
        elif isinstance(value, Morphology):
            self[key] = value
        else:
            raise TypeError(('Cannot create a new subtree "%s" for an object '
                             'of type %s.') % (key, type(value)))

    def __len__(self):
        """
        Returns the total number of compartments.
        """
        return self.n + sum(len(child) for child in self.children)

    def _update_indices(self, origin=0):
        self._origin = origin
        n = self.n
        for child in self.children:
            child._update_indices(origin=origin + n)
            n += len(child)

    def plot(self, axes=None, simple=True, origin=None):
        """
        Plots the morphology in 3D. Units are um.

        Parameters
        ----------
        axes : `Axes3D`
            the figure axes (new figure if not given)
        simple : bool, optional
            if ``True``, the diameter of branches is ignored
            (defaults to ``True``)
        """
        try:
            from pylab import figure
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError('matplotlib 0.99.1 is required for 3d plots')
        if axes is None:  # new figure
            fig = figure()
            axes = Axes3D(fig)
        x, y, z, d = self.x / um, self.y / um, self.z / um, self.diameter / um
        if origin is not None:
            x0, y0, z0 = origin
            x = hstack((x0, x))
            y = hstack((y0, y))
            z = hstack((z0, z))
        if len(x) == 1:  # root with a single compartment: probably just the soma
            axes.plot(x, y, z, "r.", linewidth=d[0])
        else:
            if simple:
                axes.plot(x, y, z, "k")
            else:  # linewidth reflects compartment diameter
                for n in range(1, len(x)):
                    axes.plot([x[n - 1], x[n]], [y[n - 1], y[n]],
                              [z[n - 1], z[n]], 'k', linewidth=d[n - 1])
        for c in self.children:
            c.plot(origin=(x[-1], y[-1], z[-1]), axes=axes, simple=simple)


class Cylinder(Morphology):
    """
    A cylinder.

    Parameters
    ----------
    length : `Quantity`, optional
        The total length in `meter`. If unspecified, inferred from `x`, `y`, `z`.
    diameter : `Quantity`
        The diameter in `meter`.
    n : int, optional
        Number of compartments (default 1).
    type : str, optional
        Type of segment, `soma`, 'axon' or 'dendrite'.
    x : `Quantity`, optional
        x position of end point in `meter` units.
        If not specified, inferred from `length` with a random direction.
    y : `Quantity`, optional
        x position of end point in `meter` units.
    z : `Quantity`, optional
        x position of end point in `meter` units.
    """

    @check_units(length=meter, diameter=meter, n=1, x=meter, y=meter, z=meter)
    def __init__(self, length=None, diameter=None, n=1, type=None, x=None,
                 y=None, z=None):
        """
        Creates a cylinder.
        n: number of compartments.
        type : 'soma', 'axon' or 'dendrite'
        x,y,z : end point (relative to origin of cylinder)
        length is optional (and ignored) if x,y,z is specified
        If x,y,z unspecified: random direction
        """
        Morphology.__init__(self, n=n)
        if x is None:
            theta = rand() * 2 * pi
            phi = rand() * 2 * pi
            x = length * sin(theta) * cos(phi)
            y = length * sin(theta) * sin(phi)
            z = length * cos(theta)
        else:
            if length is not None:
                raise AttributeError(('Length and x-y-z coordinates cannot '
                                      'be simultaneously specified'))
            length = sqrt(x**2 + y**2 + z**2)
        scale = arange(1, n + 1) * 1. / n
        self._x, self._y, self._z = x * scale, y * scale, z * scale
        self._length = ones(n) * length / n
        self._diameter = ones(n) * diameter
        self._distance = cumsum(self.length)
        self._update_area()
        self.type = type


class Soma(Morphology):  # or Sphere?
    """
    A spherical soma.

    Parameters
    ----------
    diameter : `Quantity`, optional
        Diameter of the sphere.
    """

    @check_units(diameter=meter)
    def __init__(self, diameter=None):
        Morphology.__init__(self, n=1)
        self.diameter = ones(1) * diameter
        self.type = 'soma'

    def _set_n(self, n):
        raise TypeError('Cannot change the number of compartments')

    def _update_area(self):
        self._area = ones(1) * pi * self.diameter ** 2


if __name__ == '__main__':
    from pylab import show

    morpho = Morphology('mp_ma_40984_gc2.CNG.swc')  # retinal ganglion cell
    print len(morpho), "compartments"
    # morpho.axon = None
    morpho.plot()
    # morpho=Cylinder(length=10*um,diameter=1*um,n=10)
    #morpho.plot(simple=True)
    morpho = Soma(diameter=10 * um)
    morpho.dendrite = Cylinder(length=3 * um, diameter=1 * um, n=10)
    morpho.dendrite.L = Cylinder(length=5 * um, diameter=1 * um, n=10)
    morpho.dendrite.R = Cylinder(length=7 * um, diameter=1 * um, n=10)
    morpho.dendrite.LL = Cylinder(length=3 * um, diameter=1 * um, n=10)
    morpho.axon = Morphology(n=5)
    morpho.axon.diameter = ones(5) * 1 * um
    morpho.axon.length = [1, 2, 1, 3, 1] * um
    morpho.plot(simple=True)
    show()
