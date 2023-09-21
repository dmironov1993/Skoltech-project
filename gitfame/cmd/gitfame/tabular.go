package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
)

func EncodeAuthorsToTabular(authors Authors) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', 0)
	defer func(w *tabwriter.Writer) {
		err := w.Flush()
		if err != nil {
			panic("Tabular: defer func generates an error")
		}
	}(w)

	_, err := fmt.Fprintln(w, "Name\tLines\tCommits\tFiles")
	if err != nil {
		return
	}

	lines := make([]string, len(authors.List))
	for i, author := range authors.List {
		lines[i] = fmt.Sprintf("%s\t%d\t%d\t%d",
			author.Name,
			author.Stats.Lines,
			author.Stats.Commits,
			author.Stats.Files,
		)
	}

	_, err = fmt.Fprintln(w, strings.Join(lines, "\n"))
	if err != nil {
		return
	}
}
