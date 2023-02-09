/**
 * Copyright (c) 2017 European Organisation for Nuclear Research (CERN), All Rights Reserved.
 */

package cern.accsoft.steering.jmad.modeldefs;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.accsoft.steering.jmad.domain.elem.Element;
import cern.accsoft.steering.jmad.domain.ex.JMadModelException;
import cern.accsoft.steering.jmad.domain.machine.SequenceDefinition;
import cern.accsoft.steering.jmad.domain.result.tfs.TfsResult;
import cern.accsoft.steering.jmad.domain.result.tfs.TfsResultRequestImpl;
import cern.accsoft.steering.jmad.domain.var.enums.MadxTwissVariable;
import cern.accsoft.steering.jmad.model.JMadModel;
import cern.accsoft.steering.jmad.model.JMadModelStartupConfiguration;
import cern.accsoft.steering.jmad.modeldefs.domain.JMadModelDefinition;
import cern.accsoft.steering.jmad.modeldefs.domain.OpticsDefinition;
import cern.accsoft.steering.jmad.service.JMadService;
import cern.accsoft.steering.jmad.service.JMadServiceFactory;

/**
 * The creation of this test was triggered by a bug appearing when moving to a newer version of xstream. At this time
 * enums in the beam object were not correctly loaded for the model definitions. To track down this bug and avoid
 * similar cases in the future, this test checks (for one element of each beam) that for B1 the sign of k1 corresponds
 * to k1l, while for B2 of LHC, k1l will have the opposite sign of k1, because of the BV flag (which is -1 in this
 * case).
 * 
 * @author delph, kfuchsbe
 */
public class BvFlagRegressionTest {

    private static final String ELEMENT_B2 = "MQT.12L7.B2";
    private static final String ELEMENT_B1 = "MQT.12L7.B1";
    private static final String ATTRIBUTE_NAME_K1 = "k1";
    private static final String SEQUENCE_B1 = "lhcb1";
    private static final String SEQUENCE_B2 = "lhcb2";

    private JMadService jmadService;

    @BeforeClass
    public static void setUpBeforeClass() {
        BasicConfigurator.configure();
    }

    @Before
    public void setUp() {
        jmadService = JMadServiceFactory.createJMadService();
    }

    @After
    public void tearDown() {
        jmadService.getModelManager().cleanup();
    }

    @Test
    public void lhcModelDefinitionIsAvailable() {
        assertThat(lhcModelDefinition()).isNotNull();
    }

    @Test
    public void lhcHasTwoSequenceDefintitions() {
        assertThat(lhcSequenceDefinitions()).hasSize(2);
    }

    @Test
    public void k1IsNegativeForB1() throws Exception {
        assertThat(k1ForSeqenceAndElement(SEQUENCE_B1, ELEMENT_B1)).isNegative();
    }

    @Test
    public void k1lIsNegativeForB1() throws JMadModelException {
        assertThat(k1lForSequenceAndElement(SEQUENCE_B1, ELEMENT_B1)).isNegative();
    }

    @Test
    public void k1IsNegativeForB2() throws Exception {
        assertThat(k1ForSeqenceAndElement(SEQUENCE_B2, ELEMENT_B2)).isNegative();
    }

    @Test
    public void k1lIsPositiveForB2() throws JMadModelException {
        assertThat(k1lForSequenceAndElement(SEQUENCE_B2, ELEMENT_B2)).isPositive();
    }

    private Double k1ForSeqenceAndElement(String sequenceName, String elementName) throws JMadModelException {
        JMadModel lhcModel = initiateModelFor(sequenceName);
        Element element = lhcModel.getActiveRange().getElement(elementName);
        return element.getAttribute(ATTRIBUTE_NAME_K1);
    }

    private Double k1lForSequenceAndElement(String sequenceName, String elementName) throws JMadModelException {
        JMadModel lhcModel = initiateModelFor(sequenceName);
        TfsResult tfsResult = twissForElement(lhcModel, elementName);
        return tfsResult.getDoubleData(MadxTwissVariable.K1L).get(tfsResult.getElementIndex(elementName));
    }

    private TfsResult twissForElement(JMadModel lhcModel, String elementName) throws JMadModelException {
        TfsResultRequestImpl request = TfsResultRequestImpl.createDefaultRequest();
        request.addElementFilter(elementName);
        request.addVariable(MadxTwissVariable.K1L);
        return lhcModel.twiss(request);
    }

    private List<SequenceDefinition> lhcSequenceDefinitions() {
        return lhcModelDefinition().getSequenceDefinitions();
    }

    private OpticsDefinition opticsDefinition() {
        return lhcModelDefinition().getOpticsDefinition("R2016a_A90C90A10mL300");
    }

    private JMadModelDefinition lhcModelDefinition() {
        return jmadService.getModelDefinitionManager().getModelDefinition("LHC 2016");
    }

    private JMadModel initiateModelFor(String sequenceName) throws JMadModelException {
        JMadModel lhcModel = jmadService.createModel(lhcModelDefinition());
        lhcModel.setStartupConfiguration(configurationForSequence(sequenceName));
        lhcModel.init();
        return lhcModel;
    }

    private JMadModelStartupConfiguration configurationForSequence(String sequenceName) {
        JMadModelStartupConfiguration configuration = new JMadModelStartupConfiguration();
        configuration.setInitialOpticsDefinition(opticsDefinition());
        configuration.setInitialRangeDefinition(
                lhcModelDefinition().getSequenceDefinition(sequenceName).getDefaultRangeDefinition());
        return configuration;
    }

}
