package main

import (
	"bufio"
	"bytes"
	"os/exec"
)

// cmd := exec.Command("myCommand", "arg1", "arg2")
// cmd.Dir = "/path/to/work/dir"
// cmd.Run()
// git blame HEAD --porcelain --sum/README.md

func Blame(repository, revision, fileName string) (lines []string) {
	args := []string{"blame", "--porcelain", revision, "--", fileName}
	cmd := exec.Command("git", args...)
	cmd.Dir = repository

	output, _ := cmd.Output()

	if len(output) == 0 {
		return
	}

	buf := bytes.Buffer{}
	buf.Write(output)
	scanner := bufio.NewScanner(&buf)

	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return
}
