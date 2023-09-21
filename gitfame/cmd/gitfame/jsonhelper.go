package main

type JSONFields struct {
	Name    string `json:"name"`
	Lines   int    `json:"lines"`
	Commits int    `json:"commits"`
	Files   int    `json:"files"`
}

func AuthorToJSON(author Author) JSONFields {
	return JSONFields{
		Name:    author.Name,
		Lines:   author.Stats.Lines,
		Commits: author.Stats.Commits,
		Files:   author.Stats.Files,
	}
}
