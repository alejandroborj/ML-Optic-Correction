package cern.accsoft.steering.jmad.modeldefs;

import cern.accsoft.steering.jmad.modeldefs.domain.JMadModelDefinition;
import cern.accsoft.steering.jmad.modeldefs.io.JMadModelDefinitionExportRequest;
import cern.accsoft.steering.jmad.modeldefs.io.ModelDefinitionPersistenceService;
import cern.accsoft.steering.jmad.modeldefs.io.impl.MadxScriptModelDefinitionPersistenceService;
import cern.accsoft.steering.jmad.modeldefs.io.impl.XmlModelDefinitionPersistenceService;
import org.apache.log4j.BasicConfigurator;

import java.io.File;

public class ModelDefCreator {

    public static void main(String[] args) {

        String destPath = "";
        if (args.length > 0) {
            destPath = args[0];
        }

        BasicConfigurator.configure();

        ModelDefinitionFactory[] factories = new ModelDefinitionFactory[]{ //
                new LhcNominalModelDefinitionFactory(),
        };

        ModelDefinitionPersistenceService xmlService = new XmlModelDefinitionPersistenceService();
        MadxScriptModelDefinitionPersistenceService madxScriptService = new MadxScriptModelDefinitionPersistenceService();

        for (ModelDefinitionFactory factory : factories) {
            JMadModelDefinition modelDefinition = factory.create();
            JMadModelDefinitionExportRequest exportRequest = JMadModelDefinitionExportRequest.allFrom(modelDefinition);
            if (destPath.isEmpty()) {
                destPath = "../";
            }
            try {
                File xmlFile = new File(destPath + "lhc.jmd.xml");
                System.out.println("Writing file '" + xmlFile.getAbsolutePath() + "'.");
                xmlService.save(exportRequest, xmlFile);

                File madxScriptDir = new File(destPath + "optics/");
                if (!madxScriptDir.isDirectory()) {
                    madxScriptDir.mkdirs();
                }
                madxScriptService.saveOpticScriptDirectory(exportRequest, madxScriptDir, //
                        mf -> "acc-models-lhc/" + mf.getName());
            } catch (Exception e) {
                System.out.println("Could not save model definition to file !");
                e.printStackTrace();
            }
        }

    }
}
