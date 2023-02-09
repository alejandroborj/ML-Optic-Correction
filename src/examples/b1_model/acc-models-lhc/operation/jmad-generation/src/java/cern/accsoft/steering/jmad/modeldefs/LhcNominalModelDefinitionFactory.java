/**
 *
 */
package cern.accsoft.steering.jmad.modeldefs;

import cern.accsoft.steering.jmad.domain.file.CallableModelFileImpl;
import cern.accsoft.steering.jmad.modeldefs.create.AbstractLhcModelDefinitionFactory;
import cern.accsoft.steering.jmad.modeldefs.create.OpticDefinitionSet;
import cern.accsoft.steering.jmad.modeldefs.create.OpticDefinitionSetBuilder;
import cern.accsoft.steering.jmad.modeldefs.domain.JMadModelDefinitionImpl;

import java.util.ArrayList;
import java.util.List;

import static cern.accsoft.steering.jmad.domain.file.CallableModelFile.ParseType.STRENGTHS;
import static cern.accsoft.steering.jmad.modeldefs.create.OpticModelFileBuilder.modelFile;

public class LhcNominalModelDefinitionFactory extends AbstractLhcModelDefinitionFactory {

    @Override
    protected void addInitFiles(JMadModelDefinitionImpl modelDefinition) {
        modelDefinition.addInitFile(new CallableModelFileImpl("toolkit/init-constants.madx"));
        modelDefinition.addInitFile(new CallableModelFileImpl("lhc.seq"));
    }

    @Override
    protected String getModelDefinitionName() {
        return "LHC 2022";
    }

    @Override
    protected List<OpticDefinitionSet> getOpticDefinitionSets() {
        List<OpticDefinitionSet> definitionSetList = new ArrayList<>();

        /* 2022 Nominal optics */
        definitionSetList.add(create2022AtsRampSqueezeOpticsSet());
        
        /* 60 deg optics */
        definitionSetList.add(create2022Inj60degOpticsSet());

        /* 2022 VdM optics */
        definitionSetList.add(create2022VdMOpticsSet());

        /* 2023 optics (for 2022 MD7003) */
        definitionSetList.add(create2023OpticsSet());

        /* IR7 squeeze (MD7203) */
        definitionSetList.add(createIR7SqueezeOpticsSet());

        return definitionSetList;
    }

    /**
     * ATS ramp and squeeze ... All at collision tune --> trimmed with knob to INJ (BP level)
     *
     * @return
     */
    private OpticDefinitionSet create2022AtsRampSqueezeOpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-ats.madx"));        
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-ats.madx"));

        // builder.addFinalCommonOpticFile(
        // modelFile("match-chroma.madx").doNotParse());

        /* ramp and squeeze to 1m in 1,5 */
        builder.addOptic("R2022a_A11mC11mA10mL10m", modelFile("strengths/ATS_Nominal/2022/ramp/ats_11m.madx"));        
        builder.addOptic("R2022a_A10mC10mA10mL10m", modelFile("strengths/ATS_Nominal/2022/ramp/ats_10m.madx"));        
        builder.addOptic("R2022a_A970cmC970cmA10mL970cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_970cm.madx"));
        builder.addOptic("R2022a_A930cmC930cmA10mL930cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_930cm.madx"));
        builder.addOptic("R2022a_A880cmC880cmA10mL880cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_880cm.madx"));
        builder.addOptic("R2022a_A810cmC810cmA10mL810cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_810cm.madx"));  
        builder.addOptic("R2022a_A700cmC700cmA10mL700cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_700cm.madx"));
        builder.addOptic("R2022a_A600cmC600cmA10mL600cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_600cm.madx"));
        builder.addOptic("R2022a_A510cmC510cmA10mL510cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_510cm.madx"));
        builder.addOptic("R2022a_A440cmC440cmA10mL440cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_440cm.madx"));
        builder.addOptic("R2022a_A370cmC370cmA10mL370cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_370cm.madx"));
        builder.addOptic("R2022a_A310cmC310cmA10mL310cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_310cm.madx"));
        builder.addOptic("R2022a_A250cmC250cmA10mL250cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_250cm.madx"));
        builder.addOptic("R2022a_A200cmC200cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_200cm.madx"));
//        builder.addOptic("R2021a_A200cmC200cmA10mL200cm_6-8TeV", modelFile("strengths/ATS_Nominal/2021-BeamTest/ats_200cm_6-8TeV.madx"));
        builder.addOptic("R2022a_A155cmC155cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_155cm.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/ramp/ats_133cm.madx"));
        builder.addOptic("R2022a_A118cmC118cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_118cm.madx"));
        builder.addOptic("R2022a_A104cmC104cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_104cm.madx"));
        builder.addOptic("R2022a_A89cmC89cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_89cm.madx"));
        builder.addOptic("R2022a_A71cmC71cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_71cm.madx"));
        builder.addOptic("R2022a_A60cmC60cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_60cm.madx"));
//        builder.addOptic("R2022a_level_A60cmC60cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_60cm_leveling.madx"));
        builder.addOptic("R2022a_A56cmC56cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_56cm.madx"));        
        builder.addOptic("R2022a_A52cmC52cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_52cm.madx"));
        builder.addOptic("R2022a_A48cmC48cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_48cm.madx"));        
        builder.addOptic("R2022a_A45cmC45cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_45cm.madx"));
        builder.addOptic("R2022a_A41cmC41cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_41cm.madx"));
        builder.addOptic("R2022a_A38cmC38cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_38cm.madx"));
        builder.addOptic("R2022a_A35cmC35cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_35cm.madx"));
        builder.addOptic("R2022a_A32cmC32cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_32cm.madx"));
        builder.addOptic("R2022a_A30cmC30cmA10mL200cm", modelFile("strengths/ATS_Nominal/2022/squeeze/ats_30cm.madx"));

        // Ballistic optics
        builder.addOptic("R2022_A11mC11mA10mL10m_ballistic145", modelFile("strengths/ATS_Nominal/2022/ramp/ats_11m.madx"), modelFile("strengths/ballistic/ballistic_ir145_arcs.madx"));
        
        return builder.build();
    }
    
    private OpticDefinitionSet create2022Inj60degOpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-non-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-non-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-non-ats.madx"));
        
        // 60 degree optics
        builder.addOptic("R2022_60deg_A11mC11mA10mL10m", modelFile("strengths/60deg/60deg.madx"));
    
        return builder.build();

    }

    private OpticDefinitionSet create2022VdMOpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-non-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-non-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-non-ats.madx"));
        
        // VdM optics
        builder.addOptic("R2022h_A11_2mC11_2mA11_2mL11_2m", modelFile("strengths/ATS_Nominal/VdM/ats_11_2.madx"));
        builder.addOptic("R2022h_A12_6mC12_6mA12_6mL12_6m", modelFile("strengths/ATS_Nominal/VdM/ats_12_6.madx"));
        builder.addOptic("R2022h_A14_2mC14_2mA14_2mL14_2m", modelFile("strengths/ATS_Nominal/VdM/ats_14_2.madx"));
        builder.addOptic("R2022h_A16mC16mA16mL16m", modelFile("strengths/ATS_Nominal/VdM/ats_16.madx"));
        builder.addOptic("R2022h_A17_8mC17_8mA17_8mL17_8m", modelFile("strengths/ATS_Nominal/VdM/ats_17_8.madx"));
        builder.addOptic("R2022h_A19_2mC19_2mA19_2mL19_8m", modelFile("strengths/ATS_Nominal/VdM/ats_19_2.madx"));
        builder.addOptic("R2022h_A19_2mC19_2mA19_2mL24m", modelFile("strengths/ATS_Nominal/VdM/ats_19_2_IP8.madx"));
    
        return builder.build();

    }
    
    private OpticDefinitionSet create2023OpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-ats.madx"));        
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-ats.madx"));

        
        // 2023 optics (2022 MD)
        builder.addOptic("R2023a_A11mC11mA10mL10m", modelFile("strengths/ATS_Nominal/2023_MD/ats_11m.madx"));
        builder.addOptic("R2023a_A10mC10mA10mL10m", modelFile("strengths/ATS_Nominal/2023_MD/ats_10m.madx"));
        builder.addOptic("R2023a_A970cmC970cmA10mL970cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_970cm.madx"));
        builder.addOptic("R2023a_A930cmC930cmA10mL930cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_930cm.madx"));
        builder.addOptic("R2023a_A880cmC880cmA10mL880cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_880cm.madx"));
        builder.addOptic("R2023a_A810cmC810cmA10mL810cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_810cm.madx"));
        builder.addOptic("R2023a_A700cmC700cmA10mL700cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_700cm.madx"));
        builder.addOptic("R2023a_A600cmC600cmA10mL600cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_600cm.madx"));
        builder.addOptic("R2023a_A510cmC510cmA10mL510cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_510cm.madx"));
        builder.addOptic("R2023a_A440cmC440cmA10mL440cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_440cm.madx"));
        builder.addOptic("R2023a_A370cmC370cmA10mL370cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_370cm.madx"));
        builder.addOptic("R2023a_A310cmC310cmA10mL310cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_310cm.madx"));
        builder.addOptic("R2023a_A250cmC250cmA10mL250cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_250cm.madx"));

        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_1", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_1.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-83", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_0-83.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-72", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_0-72.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-65", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_0-65.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-57", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_0-57.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-5", modelFile("strengths/ATS_Nominal/2023_MD/ats_200cm_0-5.madx"));

        builder.addOptic("R2023a_A156cmC156cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_156cm.madx"));
        builder.addOptic("R2023a_A120cmC120cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_120cm.madx"));
        builder.addOptic("R2023a_A112cmC112cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_112cm.madx"));
        builder.addOptic("R2023a_A105cmC105cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_105cm.madx"));
        builder.addOptic("R2023a_A99cmC99cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_99cm.madx"));
        builder.addOptic("R2023a_A93cmC93cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_93cm.madx"));
        builder.addOptic("R2023a_A87cmC87cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_87cm.madx"));
        builder.addOptic("R2023a_A82cmC82cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_82cm.madx"));
        builder.addOptic("R2023a_A77cmC77cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_77cm.madx"));
        builder.addOptic("R2023a_A72cmC72cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_72cm.madx"));
        builder.addOptic("R2023a_A68cmC68cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_68cm.madx"));
        builder.addOptic("R2023a_A64cmC64cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_64cm.madx"));
        builder.addOptic("R2023a_A60cmC60cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_60cm.madx"));
        builder.addOptic("R2023a_A56cmC56cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_56cm.madx"));
        builder.addOptic("R2023a_A52cmC52cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_52cm.madx"));
        builder.addOptic("R2023a_A48cmC48cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_48cm.madx"));
        builder.addOptic("R2023a_A45cmC45cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_45cm.madx"));
        builder.addOptic("R2023a_A41cmC41cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_41cm.madx"));
        builder.addOptic("R2023a_A38cmC38cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_38cm.madx"));
        builder.addOptic("R2023a_A35cmC35cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_35cm.madx"));
        builder.addOptic("R2023a_A32cmC32cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_32cm.madx"));
        builder.addOptic("R2023a_A30cmC30cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_30cm.madx"));

//        builder.addOptic("R2023a_A28cmC28cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_28cm.madx"));
//        builder.addOptic("R2023a_A26cmC26cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_26cm.madx"));
//        builder.addOptic("R2023a_A24cmC24cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_24cm.madx"));
//        builder.addOptic("R2023a_A22cmC22cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_22cm.madx"));
//        builder.addOptic("R2023a_A20cmC20cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023_MD/ats_20cm.madx"));
        
        return builder.build();
    }
    
    private OpticDefinitionSet createIR7SqueezeOpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-ats.madx"));        
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-ats.madx"));
        
        // IR7 squeeze optics
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_1", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm1.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_2", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm2.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_3", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm3.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_4", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm4.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_5", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm5.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_6", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm6.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_7", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm7.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_8", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm8.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_9", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm9.madx"));
        builder.addOptic("R2022a_A133cmC133cmA10mL200cm_IR7_10", modelFile("strengths/ATS_Nominal/IR7_Squeeze_MD/ats_133cm10.madx"));
    
        return builder.build();

    }
    

}