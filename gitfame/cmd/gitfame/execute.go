package main

func Execute() {
	ParseArguments()
	LsTree(repository, revision)
	FileLanguageExtension(filesLsTree, languages, extensions, jsonPath)
	ExclusionChecker(fileNames, exclude, restrict)
	AuthorStatistics(hashCounts, alterationCounts, fileCounts, fileNames, repository, revision, committer)
	Sort(authors, orderby)
	ValidateFormat(format)
	Format(authorSorted, format)
}
