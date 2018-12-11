#lang racket

(require racket/file)
(require racket/string)
 (require racket/flonum)



(define (parse-settings filename)
  (filter (λ (line)
            (string-prefix? line
                            ;; "Setting:"
                            "python3"))
          (file->lines filename)))

(define (parse-results filename)
  (map (lambda (line)
         (map (lambda (x)
                (real->decimal-string
                 (* (string->number x) 100) 1))
              (regexp-match* #rx"[0-9]+\\.[0-9]+"
                             line)))
       (filter (lambda (line)
                 (string-prefix? line "Test result"))
               (file->lines filename))))

(define (parse-mutate filename)
  (let ([settings (parse-settings filename)]
        [results (parse-results filename)])
    (map (λ (setting res)
           (list setting
                 (last res)))
         settings results)))

(define (parse-mutate-result-only filename)
  (let ([settings (parse-settings filename)]
        [results (parse-results filename)])
    (map (λ (setting res)
           (last res))
         settings results)))

(define (parse-negative filename)
  (let ([settings (parse-settings filename)]
        [results (parse-results filename)])
    (map (λ (setting res)
           (list
            setting
            (second res)))
         settings results)))

(define (parse-negative-result-only filename)
  (let ([settings (parse-settings filename)]
        [results (parse-results filename)])
    (map (λ (setting res)
           (second res))
         settings results)))

(module+ test

  (parse-negative "log/5000-negative.txt")
  (parse-mutate "log/5000-mutate.txt")
  
  (parse-negative "log/30000-negative.txt")
  (parse-mutate "log/30000-mutate.txt")
  (parse-negative "log/12-9-30000-negative.txt")
  (parse-mutate "log/12-9-30000-mutate.txt")
  

  

  (regexp-match* #rx"[0-9]+\\.[0-9]+"
                 "Test result:  [0.036441526412963866, 0.1510746046304703, 0.036441526412963866, 0.5, 0.8433566784858704]")
  
  )

