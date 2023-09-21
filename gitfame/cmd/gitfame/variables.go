package main

var (
	repository string
	revision   string
	orderby    string
	committer  bool
	format     string
	extensions []string
	languages  []string
	exclude    []string
	restrict   []string

	filesLsTree []string
	jsonPath    = "../../configs/language_extensions.json"
	fileNames   []string
	authors     = make(map[string]*Stats)

	hashCounts       = make(map[string]map[string]struct{})
	alterationCounts = make(map[string]int)
	fileCounts       = make(map[string]int)

	authorSorted Authors
)
