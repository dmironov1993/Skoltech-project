package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// ./gitframe/configs/language_extensions.json
// https://www.sohamkamani.com/golang/json/

// parsing (unmarshalling) language pure json file in json array
// inputFiles contains file names parsed by LsTree
// language flags are case-insensitive
// outputFiles contains file names with extensions corresponding to a language

type LanguageExtension struct {
	Name       string
	Type       string
	Extensions []string
}

func ParseJSONFromFile(jsonPath string) ([]LanguageExtension, error) {
	jsonFile, err := os.Open(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %v", err)
	}
	defer func(jsonFile *os.File) {
		err = jsonFile.Close()
		if err != nil {
			panic(err)
		}
	}(jsonFile)

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var languageExtensions []LanguageExtension
	err = json.Unmarshal(byteValue, &languageExtensions)
	if err != nil {
		return nil, fmt.Errorf("cannot unmarshal JSON: %v", err)
	}

	return languageExtensions, nil
}

func GenerateExtensionMap(languageExtensions []LanguageExtension) map[string][]string {
	extensionMap := make(map[string][]string, len(languageExtensions))
	for _, extension := range languageExtensions {
		extensionMap[strings.ToLower(extension.Name)] = extension.Extensions
	}

	return extensionMap
}

func GenerateLanguageMap(languages []string, extensionMap map[string][]string) map[string]bool {
	languageMap := make(map[string]bool)
	for _, lang := range languages {
		if extensionList, ok := extensionMap[lang]; ok {
			for _, ext := range extensionList {
				languageMap[ext] = true
			}
		}
	}

	return languageMap
}

func GenerateExtensionsSet(extensions []string) map[string]bool {
	extensionsSet := make(map[string]bool, len(extensions))
	for _, ext := range extensions {
		extensionsSet[ext] = true
	}

	return extensionsSet
}

func FileLanguageExtension(inputFiles, languages, extensions []string, jsonPath string) {
	langExtensions, err := ParseJSONFromFile(jsonPath)
	if err != nil {
		panic(err)
	}

	extensionMap := GenerateExtensionMap(langExtensions)

	languageMap := GenerateLanguageMap(languages, extensionMap)

	extensionsSet := GenerateExtensionsSet(extensions)

	for _, fileName := range inputFiles {
		fileExt := filepath.Ext(fileName)
		if (len(languageMap) == 0 || languageMap[fileExt]) && (len(extensionsSet) == 0 || extensionsSet[fileExt]) {
			fileNames = append(fileNames, fileName)
		}
	}
}
