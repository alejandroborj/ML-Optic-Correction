package cern.accsoft.steering.jmad.modeldefs.create;


import cern.accsoft.steering.jmad.MadXConstants;
import cern.accsoft.steering.jmad.domain.beam.Beam;

public final class BeamFactory {

	private BeamFactory() {
		throw new UnsupportedOperationException("only static methods");
	}
	
	/**
	 * creates the beam which can be used by default for LHC sequences
	 * 
	 * @return the beam
	 */
	public static Beam createDefaultLhcBeam() {
		double energy = 450.0; // energy in GeV
		double gamma = energy / MadXConstants.MASS_PROTON;
		double emittance = 3.5e-06; // normalized emittance
		double xEmittance = emittance / (gamma);
		double yEmittance = emittance / (gamma);

		Beam beam = new Beam();
		beam.setParticle(Beam.Particle.PROTON);
		beam.setEnergy(energy);
		beam.setBunchLength(0.077);
		beam.setDirection(Beam.Direction.PLUS);
		beam.setParticleNumber(1.1E11);
		beam.setRelativeEnergySpread(5e-4);
		beam.setHorizontalEmittance(xEmittance);
		beam.setVerticalEmittance(yEmittance);
		return beam;
	}
}
