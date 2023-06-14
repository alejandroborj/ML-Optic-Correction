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
        return "LHC 2023";
    }

    @Override
    protected List<OpticDefinitionSet> getOpticDefinitionSets() {
        List<OpticDefinitionSet> definitionSetList = new ArrayList<>();

//        /* 2022 VdM optics */
//        definitionSetList.add(create2022VdMOpticsSet());

        /* 2023 injection optics with Phase knob 25% */
        definitionSetList.add(createInjectionOptics_25Knob());

        /* 2023 injection optics with Phase knob 50% */
        definitionSetList.add(createInjectionOptics_50Knob());

        /* 2023 injection optics with Phase knob 75% */
        definitionSetList.add(createInjectionOptics_75Knob());

        /* 2023 injection optics with Phase knob 100% */
        definitionSetList.add(createInjectionOptics_100Knob());

        /* 2023 low beta optics */
        definitionSetList.add(createLowBetaOpticsSet());

//        /* IONS optics */
        definitionSetList.add(createIR2SqueezeIonsOpticsSet());

//        /* HB optics */
        definitionSetList.add(createHBOpticsSet());

        return definitionSetList;
    }

    /**
     * ATS ramp and squeeze ... All at collision tune --> trimmed with knob to INJ (BP level)
     * @return
     */

    private OpticDefinitionSet createInjectionOptics_25Knob() {
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

        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-phasechange-knobs.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/set-phasechange-knob-25strength.madx"));

        // 2023 optics
        builder.addOptic("R2023a_A11mC11mA10mL10m_PhaseKnob25ON", modelFile("strengths/ATS_Nominal/2023/ats_11m.madx"));

        return builder.build();
    }

    private OpticDefinitionSet createInjectionOptics_50Knob() {
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

        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-phasechange-knobs.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/set-phasechange-knob-50strength.madx"));

        // 2023 optics
        builder.addOptic("R2023a_A11mC11mA10mL10m_PhaseKnob50ON", modelFile("strengths/ATS_Nominal/2023/ats_11m.madx"));

        return builder.build();
    }
    private OpticDefinitionSet createInjectionOptics_75Knob() {
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

        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-phasechange-knobs.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/set-phasechange-knob-75strength.madx"));

        // 2023 optics
        builder.addOptic("R2023a_A11mC11mA10mL10m_PhaseKnob75ON", modelFile("strengths/ATS_Nominal/2023/ats_11m.madx"));

        return builder.build();
    }
    private OpticDefinitionSet createInjectionOptics_100Knob() {
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
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-phasechange-knobs.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/set-phasechange-knob-100strength.madx"));

        // 2023 optics
        builder.addOptic("R2023a_A11mC11mA10mL10m_PhaseKnob100ON", modelFile("strengths/ATS_Nominal/2023/ats_11m.madx"));

        return builder.build();
    }


    private OpticDefinitionSet createLowBetaOpticsSet() {
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
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-phasechange-knobs.madx"));

        // 2023 optics
        builder.addOptic("R2023a_A11mC11mA10mL10m", modelFile("strengths/ATS_Nominal/2023/ats_11m.madx"));
        builder.addOptic("R2023a_A10mC10mA10mL10m", modelFile("strengths/ATS_Nominal/2023/ats_10m.madx"));
        builder.addOptic("R2023a_A970cmC970cmA10mL970cm", modelFile("strengths/ATS_Nominal/2023/ats_970cm.madx"));
        builder.addOptic("R2023a_A930cmC930cmA10mL930cm", modelFile("strengths/ATS_Nominal/2023/ats_930cm.madx"));
        builder.addOptic("R2023a_A880cmC880cmA10mL880cm", modelFile("strengths/ATS_Nominal/2023/ats_880cm.madx"));
        builder.addOptic("R2023a_A810cmC810cmA10mL810cm", modelFile("strengths/ATS_Nominal/2023/ats_810cm.madx"));
        builder.addOptic("R2023a_A700cmC700cmA10mL700cm", modelFile("strengths/ATS_Nominal/2023/ats_700cm.madx"));
        builder.addOptic("R2023a_A600cmC600cmA10mL600cm", modelFile("strengths/ATS_Nominal/2023/ats_600cm.madx"));
        builder.addOptic("R2023a_A510cmC510cmA10mL510cm", modelFile("strengths/ATS_Nominal/2023/ats_510cm.madx"));
        builder.addOptic("R2023a_A440cmC440cmA10mL440cm", modelFile("strengths/ATS_Nominal/2023/ats_440cm.madx"));
        builder.addOptic("R2023a_A370cmC370cmA10mL370cm", modelFile("strengths/ATS_Nominal/2023/ats_370cm.madx"));
        builder.addOptic("R2023a_A310cmC310cmA10mL310cm", modelFile("strengths/ATS_Nominal/2023/ats_310cm.madx"));
        builder.addOptic("R2023a_A250cmC250cmA10mL250cm", modelFile("strengths/ATS_Nominal/2023/ats_250cm.madx"));

        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_1", modelFile("strengths/ATS_Nominal/2023/ats_200cm_1.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-83", modelFile("strengths/ATS_Nominal/2023/ats_200cm_0-83.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-72", modelFile("strengths/ATS_Nominal/2023/ats_200cm_0-72.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-65", modelFile("strengths/ATS_Nominal/2023/ats_200cm_0-65.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-57", modelFile("strengths/ATS_Nominal/2023/ats_200cm_0-57.madx"));
        builder.addOptic("R2023a_A200cmC200cmA10mL200cm_0-5", modelFile("strengths/ATS_Nominal/2023/ats_200cm_0-5.madx"));

        builder.addOptic("R2023a_A156cmC156cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_156cm.madx"));
        builder.addOptic("R2023a_A120cmC120cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_120cm.madx"));
        builder.addOptic("R2023a_A112cmC112cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_112cm.madx"));
        builder.addOptic("R2023a_A105cmC105cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_105cm.madx"));
        builder.addOptic("R2023a_A99cmC99cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_99cm.madx"));
        builder.addOptic("R2023a_A93cmC93cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_93cm.madx"));
        builder.addOptic("R2023a_A87cmC87cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_87cm.madx"));
        builder.addOptic("R2023a_A82cmC82cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_82cm.madx"));
        builder.addOptic("R2023a_A77cmC77cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_77cm.madx"));
        builder.addOptic("R2023a_A72cmC72cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_72cm.madx"));
        builder.addOptic("R2023a_A68cmC68cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_68cm.madx"));
        builder.addOptic("R2023a_A64cmC64cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_64cm.madx"));
        builder.addOptic("R2023a_A60cmC60cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_60cm.madx"));
        builder.addOptic("R2023a_A56cmC56cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_56cm.madx"));
        builder.addOptic("R2023a_A52cmC52cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_52cm.madx"));
        builder.addOptic("R2023a_A48cmC48cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_48cm.madx"));
        builder.addOptic("R2023a_A45cmC45cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_45cm.madx"));
        builder.addOptic("R2023a_A41cmC41cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_41cm.madx"));
        builder.addOptic("R2023a_A38cmC38cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_38cm.madx"));
        builder.addOptic("R2023a_A35cmC35cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_35cm.madx"));
        builder.addOptic("R2023a_A32cmC32cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_32cm.madx"));
        builder.addOptic("R2023a_A30cmC30cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_30cm.madx"));

//        builder.addOptic("R2023a_A28cmC28cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_28cm.madx"));
//        builder.addOptic("R2023a_A26cmC26cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_26cm.madx"));
//        builder.addOptic("R2023a_A24cmC24cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_24cm.madx"));
//        builder.addOptic("R2023a_A22cmC22cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_22cm.madx"));
//        builder.addOptic("R2023a_A20cmC20cmA10mL200cm", modelFile("strengths/ATS_Nominal/2023/ats_20cm.madx"));
        
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

    private OpticDefinitionSet createHBOpticsSet() {
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

        // HB optics to 120m (starting from 17_8 (VdM))
        builder.addOptic("R2023h_A24mC24mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/24m.madx"));
        builder.addOptic("R2023h_A36mC36mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/36m.madx"));
        builder.addOptic("R2023h_A52mC52mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/52m.madx"));
        builder.addOptic("R2023h_A75mC75mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/75m.madx"));
        builder.addOptic("R2023h_100mC100mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/100m.madx"));
        builder.addOptic("R2023h_A112mC112mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/112m.madx"));
        builder.addOptic("R2023h_A120mC120mA18L18", modelFile("strengths/ATS_Nominal/2023_HB/120m.madx"));

        return builder.build();
    }

    private OpticDefinitionSet createIR2SqueezeIonsOpticsSet() {
        OpticDefinitionSetBuilder builder = OpticDefinitionSetBuilder.newInstance();

        /* initial optics strength files common to all optics (loaded before the other strength files) */
        builder.addInitialCommonOpticFile(modelFile("toolkit/zero-strengths.madx"));
        /* final optics strength files common to all optics (loaded after the other strength files) */
        builder.addFinalCommonOpticFile(modelFile("toolkit/reset-bump-flags.madx").parseAs(STRENGTHS));
        builder.addFinalCommonOpticFile(modelFile("toolkit/match-lumiknobs.madx"));
        /* Define correct knobs to be used operationally */
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-tune-knobs-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-chroma-knobs-ats.madx"));
        builder.addFinalCommonOpticFile(modelFile("toolkit/generate-op-coupling-knobs-non-ats.madx"));
        
        // ION test optics
        builder.addOptic("R2023i_A11mC11mA10mL10m", modelFile("strengths/ATS_Nominal/2023_IONS/11_10m.madx"));
        builder.addOptic("R2023i_A970cmC970cmA970cmL970cm", modelFile("strengths/ATS_Nominal/2023_IONS/970cm.madx"));
        builder.addOptic("R2023i_A920cmC920cmA920cmL920cm", modelFile("strengths/ATS_Nominal/2023_IONS/920cm.madx"));
        builder.addOptic("R2023i_A850cmC850cmA850cmL850cm", modelFile("strengths/ATS_Nominal/2023_IONS/850cm.madx"));
        builder.addOptic("R2023i_A760cmC760cmA760cmL760cm", modelFile("strengths/ATS_Nominal/2023_IONS/760cm.madx"));
        builder.addOptic("R2023i_A670cmC670cmA670cmL670cm", modelFile("strengths/ATS_Nominal/2023_IONS/670cm.madx"));
        builder.addOptic("R2023i_A590cmC590cmA590cmL590cm", modelFile("strengths/ATS_Nominal/2023_IONS/590cm.madx"));
        builder.addOptic("R2023i_A520cmC520cmA520cmL520cm", modelFile("strengths/ATS_Nominal/2023_IONS/520cm.madx"));
        builder.addOptic("R2023i_A450cmC450cmA450cmL450cm", modelFile("strengths/ATS_Nominal/2023_IONS/450cm.madx"));
        builder.addOptic("R2023i_A400cmC400cmA400cmL400cm", modelFile("strengths/ATS_Nominal/2023_IONS/400cm.madx"));
        builder.addOptic("R2023i_A360cmC360cmA360cmL360cm", modelFile("strengths/ATS_Nominal/2023_IONS/360cm.madx"));
        builder.addOptic("R2023i_A320cmC320cmA320cmL320cm", modelFile("strengths/ATS_Nominal/2023_IONS/320cm.madx"));
        builder.addOptic("R2023i_A290cmC290cmA290cmL290cm", modelFile("strengths/ATS_Nominal/2023_IONS/290cm.madx"));
        builder.addOptic("R2023i_A230cmC230cmA230cmL230cm", modelFile("strengths/ATS_Nominal/2023_IONS/230cm.madx"));
        builder.addOptic("R2023i_A185cmC185cmA185cmL185cm", modelFile("strengths/ATS_Nominal/2023_IONS/185cm.madx"));
        builder.addOptic("R2023i_A135cmC135cmA135cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/135cm.madx"));
        builder.addOptic("R2023i_A100cmC100cmA100cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/100cm.madx"));
        builder.addOptic("R2023i_A82cmC82cmA82cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/82cm.madx"));
        builder.addOptic("R2023i_A68cmC68cmA68cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/68cm.madx"));
        builder.addOptic("R2023i_A57cmC57cmA57cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/57cm.madx"));
        builder.addOptic("R2023i_A50cmC50cmA50cmL150cm", modelFile("strengths/ATS_Nominal/2023_IONS/50cm.madx"));
    
        return builder.build();
    }
}