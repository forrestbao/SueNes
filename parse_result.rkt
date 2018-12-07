#lang racket

(require racket/file)
(require racket/string)

(module+ test
  (filter (lambda (line)
            (or (string-prefix? line "SETTING:")
                (string-prefix? line "RESULT:")))
          (file->lines "log.txt"))
  )

