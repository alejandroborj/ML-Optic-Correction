package cern.accsoft.steering.jmad.modeldefs;

import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;

import cern.accsoft.steering.jmad.domain.ex.JMadModelException;
import cern.accsoft.steering.jmad.domain.optics.Optic;
import cern.accsoft.steering.jmad.model.JMadModel;
import cern.accsoft.steering.jmad.model.JMadModelStartupConfiguration;
import cern.accsoft.steering.jmad.modeldefs.domain.JMadModelDefinition;
import cern.accsoft.steering.jmad.modeldefs.domain.OpticsDefinition;
import cern.accsoft.steering.jmad.service.JMadService;
import cern.accsoft.steering.jmad.service.JMadServiceFactory;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;

/**
 * This is a JUnit4 test case, that loops through all available models and checks several simple conditions.
 *
 * @author Kajetan Fuchsberger (kajetan.fuchsberger at cern.ch)
 */
@RunWith(JUnitParamsRunner.class)
public class AllModelsTest {

    private static final Logger LOGGER = Logger.getLogger(AllModelsTest.class);

    @BeforeClass
    public static void initLog4J() {
        BasicConfigurator.configure();
        Logger.getRootLogger().setLevel(Level.INFO);
    }

    @AfterClass
    public static void resetLog4J() {
        BasicConfigurator.resetConfiguration();
    }

    /** The service to get all the definitions and to create the models */
    private static final JMadService JMAD_SERVICE = JMadServiceFactory.createJMadService();

    /**
     * provides the parameters for the tests
     *
     * @return all model definitions as parameters for JUnit
     */
    public static final Collection<Object> getModelDefinitions() {
        List<Object> parameterArrays = new ArrayList<Object>();
        for (JMadModelDefinition definition : JMAD_SERVICE.getModelDefinitionManager().getAllModelDefinitions()) {
            for (OpticsDefinition opticsDefinition : definition.getOpticsDefinitions()) {
                parameterArrays.add(new Object[] { definition , opticsDefinition});
            }
        }
        return parameterArrays;
    }

    /*
     * Test methods
     */

    /**
     * Simply tests if opening the model is possible. Then it closes it again
     *
     * @throws JMadModelException if the model-creation fails
     */
    @Parameters(method = "getModelDefinitions")
    @Test
    public void testOpenModel(JMadModelDefinition modelDefinition, OpticsDefinition opticsDefinition) throws JMadModelException {
        String modelName = modelDefinition + " | " + opticsDefinition;

        LOGGER.info("");
        LOGGER.info("==== [START] Testing model '" + modelName + "'. ====");
        assertNotNull("Model definition must not be null", modelDefinition);

        /* create the model */
        JMadModel model = JMAD_SERVICE.createModel(modelDefinition);
        assertNotNull("The created model must not be null.", model);
        LOGGER.info("Model '" + model + "' successfully created.");

        /* init the model */
        JMadModelStartupConfiguration startup = new JMadModelStartupConfiguration();
        startup.setInitialOpticsDefinition(opticsDefinition);

        model.init();
        LOGGER.info("Model '" + model + "' successfully initialized.");

        Optic optic = model.getOptics();
        assertNotNull("Optic must not be null", optic);
        LOGGER.info("Optics values for optic '" + opticsDefinition + "' successfully retrieved.");

        /* and close it again */
        model.cleanup();
        LOGGER.info("Model '" + model + "' successfully cleaned.");

        /* and remove it from the manager */
        JMAD_SERVICE.getModelManager().removeModel(model);
        LOGGER.info("==== [FINISHED] Testing model '" + modelName + "'. ====");
        LOGGER.info("");
    }
}
