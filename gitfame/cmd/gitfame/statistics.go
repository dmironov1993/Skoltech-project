package main

import (
	"log"
	"strconv"
	"strings"
)

// uniqueHashCounts := make(map[string]map[string]struct{})
// totalAlterationsCounts := make(map[string]int)
// filesCounts := make(map[string]int)

// number of lines per author
// number of commits per author
// number of files per author

// git blame --porcelain
// outputs some blocks of information with a blank line in-between them
// machine text in format 'commit-hash a b c'
// commit-hash - hash information of a commit
// a - line number in initial files e.g. before changes
// b - final line number in changed file e.g. after changes
// c - count of lines that were altered

// example
// git blame --porcelain -- sum/README.md | less

/*
	d338fc2d6db9f441acbeca8f742daaa873e25241 1 1 4
	author Fedor Korotkiy
	author-mail <prime@yandex-team.ru>
	author-time 1581553619
	author-tz +0000
	committer Fedor Korotkiy
	committer-mail <prime@yandex-team.ru>
	committer-time 1581553619
	committer-tz +0000
	summary export public files
	filename sum/README.md
		# sum
	d338fc2d6db9f441acbeca8f742daaa873e25241 2 2

	d338fc2d6db9f441acbeca8f742daaa873e25241 3 3
		В этой задаче вам нужно научиться сдавать решения в тестовую систему.
	d338fc2d6db9f441acbeca8f742daaa873e25241 4 4

	bf5708bc0cd43d36e3f3cba860f9c7dc2b8f7164 5 5 2
	author Fedor Korotkiy
	author-mail <prime@yandex-team.ru>
	author-time 1581685020
	author-tz +0000
	committer Fedor Korotkiy
	committer-mail <prime@yandex-team.ru>
	committer-time 1581685020
	committer-tz +0000
	summary export public files
	previous 81d5767188b823076aa551dd4b94c7d51b7c6c45 sum/README.md
	filename sum/README.md
		0. Склонируйте этот репозиторий с https://gitlab.com/slon/shad-go
	bf5708bc0cd43d36e3f3cba860f9c7dc2b8f7164 6 6

	d338fc2d6db9f441acbeca8f742daaa873e25241 5 7 8
		1. Откройте файл `sum.go` и реализуйте функцию сложения двух чисел.
	d338fc2d6db9f441acbeca8f742daaa873e25241 6 8

	d338fc2d6db9f441acbeca8f742daaa873e25241 7 9
		2. Проверьте, что ваше решение проходит тесты локально.
	d338fc2d6db9f441acbeca8f742daaa873e25241 8 10

	d338fc2d6db9f441acbeca8f742daaa873e25241 9 11
		```shell
	d338fc2d6db9f441acbeca8f742daaa873e25241 10 12
			   # Из корня репозитория.
	d338fc2d6db9f441acbeca8f742daaa873e25241 11 13
			   go test ./sum/...
	d338fc2d6db9f441acbeca8f742daaa873e25241 12 14
			   ```
	bf5708bc0cd43d36e3f3cba860f9c7dc2b8f7164 15 15 7

*/

// lines example in column with --porcelain format
// <hash> a b c
// author ...
// author-mail ...
// author-time ...
// author-tz ...
// committer ...
// committer-mail ...
// committer-time ...
// committer-tz ...
// <empty line>
// ...
// <repeat total c times>
// ....
// <hash> a1 b1 c1
// author ...
// ....

func parseLine(line string, authorInfo map[string][]string, hashCounts map[string]int,
	currentHash *string, cntAlterations *int, committer bool) {
	lineSplit := strings.Split(line, " ")

	switch {
	case *cntAlterations == 0:
		*currentHash = lineSplit[0]
		var err error
		*cntAlterations, err = strconv.Atoi(lineSplit[len(lineSplit)-1])
		if err != nil {
			log.Printf("Error converting string to integer: %v", err)
			return
		}
		hashCounts[*currentHash] += *cntAlterations
	case lineSplit[0] == "author" && !committer:
		authorInfo[line[7:]] = append(authorInfo[line[7:]], *currentHash)
	case lineSplit[0] == "committer" && committer:
		authorInfo[line[10:]] = append(authorInfo[line[10:]], *currentHash)
	case line[0] == '\t':
		*cntAlterations--
	}
}

func updateAuthorStatistics(authorInfo map[string][]string, hashCounts map[string]int,
	uniqueHashCounts map[string]map[string]struct{}, filesCounts map[string]int, totalAlterationsCounts map[string]int) {
	for author, hashes := range authorInfo {
		filesCounts[author]++
		for _, hash := range hashes {
			if _, ok := uniqueHashCounts[author]; !ok {
				uniqueHashCounts[author] = make(map[string]struct{})
			}
			uniqueHashCounts[author][hash] = struct{}{}
			totalAlterationsCounts[hash] += hashCounts[hash]
		}
	}
}

func InitializeStatsForEmptyLines(lines []string, repository, revision, file string) (map[string]int, map[string][]string) {
	commitCounts := make(map[string]int)
	authorDetails := make(map[string][]string)

	if len(lines) == 0 {
		NoLines(repository, revision, file, &authorDetails)
	}

	return commitCounts, authorDetails
}

func NoLines(repository, revision, file string, authorDetails *map[string][]string) {
	hash, author := Log(repository, revision, file)
	(*authorDetails)[author] = append((*authorDetails)[author], hash)
}

func ParseLines(lines []string, commitCounts map[string]int, authorDetails map[string][]string, isCommitter bool) {
	currentHash := ""
	editCount := 0
	for _, line := range lines {
		parseLine(line, authorDetails, commitCounts, &currentHash, &editCount, isCommitter)
	}
}

func AuthorStatistics(uniqueCommitCounts map[string]map[string]struct{}, totalEditsCounts map[string]int,
	fileStats map[string]int, fileNames []string, repository, revision string, isCommitter bool) {

	for _, file := range fileNames {
		lines := Blame(repository, revision, file)
		commitCounts, authorDetails := InitializeStatsForEmptyLines(lines, repository, revision, file)

		ParseLines(lines, commitCounts, authorDetails, isCommitter)

		updateAuthorStatistics(authorDetails, commitCounts, uniqueCommitCounts, fileStats, totalEditsCounts)
	}

	for i, j := range hashCounts {
		lines := 0
		for k := range j {
			lines += alterationCounts[k]
		}

		authors[i] = &Stats{
			Lines:   lines,
			Commits: len(j),
			Files:   fileCounts[i],
		}
	}
}
