import jpype
from jpype import *
import os

if __name__ == "__main__":
    jvmPath = 'C:\\Program Files\\Java\\jdk-13.0.1\\bin\\server\\jvm.dll'
    jarPath = 'F:\\Dataset\\nyt_corpus\\tools\\build\\timestools.jar'
    inputPath = 'F:\\Dataset\\nyt_corpus\\data'
    outputPath = 'F:\\Dataset\\nyt_corpus\\converted'

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    jpype.startJVM(jvmPath, "-Djava.class.path="+jarPath)

    NYTCorpusDocumentParser = JClass('com.nytlabs.corpus.NYTCorpusDocumentParser')
    File = JClass('java.io.File')
    parser = NYTCorpusDocumentParser()

    cnt = 0
    for root, dirs, files in os.walk(inputPath):
        for fileName in files:
            filePath = os.path.join(root, fileName)
            if not filePath.endswith("xml"):
                continue

            document = parser.parseNYTCorpusDocumentFromFile(File(filePath), False)
            articleAbstract = document.getArticleAbstract()
            body = document.getBody()
            guid = document.getGuid()
            if articleAbstract == None or body == None or guid == None:
                continue

            outputFile = os.path.join(outputPath, str(guid)+'.story')
            with open(outputFile, "w", encoding="utf-8") as writer:
                writer.write(body + "\n\n@highlight\n\n" + articleAbstract)

            cnt += 1
            if cnt % 1000 == 0:
                print("Converted %d" % cnt)

    print("Total %d" % cnt)
    jpype.shutdownJVM()