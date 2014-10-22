import unittest
import numpy as np
from constrained_integrator import ConstrainedIntegrator

class TestConstrainedIntegrator(unittest.TestCase):

  def setUp(self):
    pass
    
  def IdentityMobility(self, x):
    mobility = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    return mobility

  def DiagonalQuadraticMobility(self, x):
    mobility = np.matrix([[1.0 + x[0, 0]**2, 0.0], [0.0, 1.0 + x[1, 0]**2]]) 
    return mobility
    
  def empty_constraint(self, x):
    return 0.0

  def test_initialize(self):
    """ Test that the integrator is set up correctly. """
    scheme = "RFD"
    initial_position = np.matrix([[0.0], [0.0]])
    
    test_integrator = ConstrainedIntegrator(
      self.empty_constraint, self.IdentityMobility, scheme, initial_position)
    # Test dimensions
    self.assertEqual(test_integrator.dim, 2)
    # Test Mobility
    self.assertEqual(test_integrator.mobility(initial_position)[0, 0], 1.0)
    self.assertEqual(test_integrator.mobility(initial_position)[1, 0], 0.0)
    self.assertEqual(test_integrator.mobility(initial_position)[0, 1], 0.0)
    self.assertEqual(test_integrator.mobility(initial_position)[1, 1], 1.0)
    # Test constraint.
    self.assertEqual(test_integrator.surface_function(10.0), 0.0)
    # Test scheme.
    self.assertEqual(test_integrator.scheme, "RFD")

  def test_scheme_check(self):
    """ Test that a nonexistant scheme isn't accepted. """
    scheme = "DOESNTEXIST"
    initial_position = np.matrix([[0.0], [0.0]])
    self.assertRaises(
      NotImplementedError,
      ConstrainedIntegrator,
      self.empty_constraint, self.IdentityMobility, scheme, initial_position)
    
  def test_normal_vector(self):
    """ Test that normal vector points in the right direction """
    scheme = 'RFD'
    initial_position = np.matrix([[1.2], [0.0]])
    def sphere_constraint(x):
      return np.sqrt(x[0,0]*x[0,0] + x[1,0]*x[1,0]) - 1.2
    
    test_integrator = ConstrainedIntegrator(
      sphere_constraint, self.IdentityMobility, scheme, initial_position)
    
    normal_vector = test_integrator.NormalVector(initial_position)
    self.assertAlmostEqual(normal_vector[0, 0], 1.0)
    self.assertAlmostEqual(normal_vector[1, 0], 0.0)
    self.assertEqual(normal_vector.shape, (2, 1))

    # Test a different position for the normal vector.
    normal_vector = test_integrator.NormalVector(
      np.matrix([[1.2/np.sqrt(2.)], [1.2/np.sqrt(2.)]]))
    self.assertAlmostEqual(normal_vector[0, 0], 1./np.sqrt(2))
    self.assertAlmostEqual(normal_vector[1, 0], 1./np.sqrt(2))
    self.assertEqual(normal_vector.shape, (2, 1))


  def test_projection_matrix(self):
    scheme = 'RFD'
    initial_position = np.matrix([[1.2], [0.0]])
    def sphere_constraint(x):
      return np.sqrt(x[0, 0]*x[0, 0] + x[1, 0]*x[1, 0]) - 1.2

    test_integrator = ConstrainedIntegrator(
      sphere_constraint, self.IdentityMobility, scheme, initial_position)

    projection_vector = test_integrator.ProjectionMatrix(initial_position)
    self.assertAlmostEqual(projection_vector[0, 0], 0.0)
    self.assertAlmostEqual(projection_vector[0, 1], 0.0)
    self.assertAlmostEqual(projection_vector[1, 0], 0.0)
    self.assertAlmostEqual(projection_vector[1, 1], 1.0)

  def test_rfd_step(self):
    ''' Test that the RFD step does the correct thing'''
    scheme = 'RFD'
    initial_position = np.matrix([[1.2], [0.0]])
    def sphere_constraint(x):
      return x[0, 0]*x[0, 0] + x[1, 0]*x[1, 0] - 1.2**2

    test_integrator = ConstrainedIntegrator(
      sphere_constraint, self.IdentityMobility, scheme, initial_position)
    test_integrator.MockRandomGenerator()
    
    test_integrator.TimeStep(0.01)
    # TODO Figure out a better way to test this:
    # self.assertAlmostEqual(test_integrator.position[1,0], np.sqrt(2)/10)
    # self.assertAlmostEqual(test_integrator.position[0,0], 1.2 - 0.03/1.2)

  def test_noise_magnitude_diagonal(self):
    ''' 
    Test that we can do the correct cholesky decomposition for
    diagonal matrices
    '''
    scheme = 'RFD'
    initial_position = np.matrix([[1.2], [0.0]])
    def sphere_constraint(x):
      return x[0, 0]*x[0, 0] + x[1, 0]*x[1, 0] - 1.2**2

    test_integrator = ConstrainedIntegrator(
      sphere_constraint, self.DiagonalQuadraticMobility, scheme, initial_position)
    
    noise_magnitude = test_integrator.NoiseMagnitude(initial_position)
    self.assertAlmostEqual(noise_magnitude[0, 0], np.sqrt(1.0 + 1.2*1.2))
    self.assertAlmostEqual(noise_magnitude[0, 1], 0.)
    self.assertAlmostEqual(noise_magnitude[1, 0], 0.)
    self.assertAlmostEqual(noise_magnitude[1, 1], 1.0)

    noise_magnitude = test_integrator.NoiseMagnitude(np.matrix([[0.84852813742385691],
                                                                [0.84852813742385691]]))
    self.assertAlmostEqual(noise_magnitude[0, 0], 1.31148770486)
    self.assertAlmostEqual(noise_magnitude[0, 1], 0.)
    self.assertAlmostEqual(noise_magnitude[1, 0], 0.)
    self.assertAlmostEqual(noise_magnitude[1, 1], 1.31148770486)

if __name__ == "__main__":
  unittest.main()
    
