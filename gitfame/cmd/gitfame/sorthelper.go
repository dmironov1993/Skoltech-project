package main

type Stats struct {
	Lines   int
	Commits int
	Files   int
}

type Author struct {
	Stats Stats
	Name  string
}

type Authors struct {
	List    []Author
	Orderby string
}
