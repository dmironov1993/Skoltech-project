package main

import "path/filepath"

func Matcher(patterns []string, fileName string) (matched bool) {
	for _, pattern := range patterns {
		if ok, _ := filepath.Match(pattern, fileName); ok {
			return true
		}
	}
	return false
}

func Checker(fileName string, exclude, restrict []string) (check bool) {
	if Matcher(exclude, fileName) {
		return false
	}

	if len(restrict) == 0 {
		return true
	}

	return Matcher(restrict, fileName)
}

func ExclusionChecker(inputFiles, exclude, restrict []string) {
	var newFiles []string
	for _, fileName := range inputFiles {
		checker := Checker(fileName, exclude, restrict)
		if checker {
			newFiles = append(newFiles, fileName)
		}
	}
	fileNames = newFiles
}
