#lang racket

(require racket/file)
(require racket/string)

(module+ test
  (filter (lambda (line)
            (or (string-prefix? line "SETTING:")
                (string-prefix? line "RESULT:")))
          (file->lines "log-negative-5000.txt"))

  (let ([settings (filter (λ (line)
                            (string-prefix? line "Setting:"))
                          (file->lines "log-mutated-5000.txt"))]
        [results (map (lambda (line)
                        (regexp-match* #rx"[0-9]+\\.[0-9]+"
                                       line))
                      (filter (lambda (line)
                                (string-prefix? line "Test result"))
                              (file->lines "log-mutated-5000.txt")))])
    (map (λ (setting res)
           (list setting
                 ;; MSE, also the loss
                 (first res)
                 ;; MAE
                 (second res)
                 ;; PCC
                 (last res)))
         settings results))

  (regexp-match* #rx"[0-9]+\\.[0-9]+"
                 "Test result:  [0.036441526412963866, 0.1510746046304703, 0.036441526412963866, 0.5, 0.8433566784858704]")
  
  )

