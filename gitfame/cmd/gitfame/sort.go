package main

import (
	"sort"
	"strings"
)

func createAuthors(a map[string]*Stats) Authors {
	var author Authors
	for name, stat := range a {
		author.List = append(author.List, Author{
			Stats: *stat,
			Name:  name,
		})
	}
	return author
}

func compareByLines(a, b Author) bool {
	if a.Stats.Lines != b.Stats.Lines {
		return a.Stats.Lines > b.Stats.Lines
	}
	if a.Stats.Commits != b.Stats.Commits {
		return a.Stats.Commits > b.Stats.Commits
	}
	if a.Stats.Files != b.Stats.Files {
		return a.Stats.Files > b.Stats.Files
	}

	return strings.ToLower(a.Name) < strings.ToLower(b.Name)
}

func compareByCommits(a, b Author) bool {
	if a.Stats.Commits != b.Stats.Commits {
		return a.Stats.Commits > b.Stats.Commits
	}
	if a.Stats.Lines != b.Stats.Lines {
		return a.Stats.Lines > b.Stats.Lines
	}
	if a.Stats.Files != b.Stats.Files {
		return a.Stats.Files > b.Stats.Files
	}

	return strings.ToLower(a.Name) < strings.ToLower(b.Name)
}

func compareByFiles(a, b Author) bool {
	if a.Stats.Files != b.Stats.Files {
		return a.Stats.Files > b.Stats.Files
	}
	if a.Stats.Lines != b.Stats.Lines {
		return a.Stats.Lines > b.Stats.Lines
	}
	if a.Stats.Commits != b.Stats.Commits {
		return a.Stats.Commits > b.Stats.Commits
	}

	return strings.ToLower(a.Name) < strings.ToLower(b.Name)
}

func sortAuthors(authors Authors, orderby string) {
	authors.Orderby = orderby
	sort.Slice(authors.List, func(i, j int) bool {
		a, b := authors.List[i], authors.List[j]

		switch orderby {
		case "lines":
			return compareByLines(a, b)
		case "commits":
			return compareByCommits(a, b)
		case "files":
			return compareByFiles(a, b)
		default:
			panic("SORT.GO: sortAuthorSLice incorrect orderby")
		}
	})
}

func updateGlobalVariable(authors Authors) {
	authorSorted = authors
}

func Sort(authors map[string]*Stats, orderby string) {
	authorList := createAuthors(authors)
	sortAuthors(authorList, orderby)
	updateGlobalVariable(authorList)
}
