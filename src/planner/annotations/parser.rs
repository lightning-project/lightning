use std::error::Error;
use std::fmt::{self, Display};
use std::ops::Range;

use super::{AccessMode, AnnotationRule, Expr, Var};
use crate::planner::annotations::{AccessPattern, SliceExpr};
use crate::prelude::*;
use crate::types::{ReductionFunction, MAX_DIMS};

fn tokenize(input: &str) -> Vec<Range<usize>> {
    let mut iter = input.char_indices().peekable();
    let mut spans = vec![];

    while let Some((start, c)) = iter.next() {
        // Ignore whitespaces
        if c.is_ascii_whitespace() {
            continue;
        }

        // If is alphanumeric, parse as identifier
        if c.is_ascii_alphabetic() {
            while let Some(_) = iter.next_if(|&(_, c)| c.is_ascii_alphanumeric() || c == '_') {
                //
            }
        } else if c.is_ascii_digit() {
            // If it is digit, parse as a number
            while let Some(_) = iter.next_if(|&(_, c)| c.is_ascii_digit()) {
                //
            }
        } else { // Not a identifier or number, simply parse as single token.
             //
        }

        let end = iter.peek().map(|&(i, _)| i).unwrap_or(input.len());
        spans.push(start..end);
    }

    spans
}

struct TokenError {
    index: usize,
}

#[derive(Clone, Copy)]
struct TokenStream<'a> {
    input: &'a str,
    spans: &'a [Range<usize>],
    cursor: usize,
}

impl<'a> TokenStream<'a> {
    fn new(input: &'a str, spans: &'a [Range<usize>]) -> Self {
        Self {
            input,
            spans,
            cursor: 0,
        }
    }

    fn peek(&mut self) -> &'a str {
        self.spans
            .get(self.cursor)
            .map(|r| &self.input[r.clone()])
            .unwrap_or_default()
    }

    fn next(&mut self) -> &'a str {
        let result = self.peek();
        self.cursor += 1;
        result
    }

    fn next_if<F>(&mut self, fun: F) -> Result<&'a str, TokenError>
    where
        F: FnOnce(&'a str) -> bool,
    {
        let token = self.peek();

        if (fun)(token) {
            self.cursor += 1;
            Ok(token)
        } else {
            Err(self.prev_error())
        }
    }

    fn consume(&mut self, expected: &'a str) -> Result<(), TokenError> {
        self.next_if(|t| t == expected).map(|_| ())
    }

    fn try_advance<F, T>(&mut self, fun: F) -> Result<T, TokenError>
    where
        F: FnOnce(&mut Self) -> Result<T, TokenError>,
    {
        let old_cursor = self.cursor;

        match (fun)(self) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.cursor = old_cursor;
                Err(e)
            }
        }
    }

    fn end(&self) -> bool {
        self.cursor >= self.spans.len()
    }

    fn prev_error(&mut self) -> TokenError {
        TokenError {
            index: self.cursor - 1,
        }
    }

    fn error(&self) -> TokenError {
        TokenError { index: self.cursor }
    }
}

fn is_ident(s: &str) -> bool {
    s.char_indices()
        .all(|(i, c)| c.is_ascii_alphabetic() || c == '_' || (i != 0 && c.is_ascii_digit()))
        && !s.is_empty()
}

fn parse_number(s: &str) -> Option<i64> {
    s.parse().ok()
}

fn parse_binding<'a>(
    tokens: &mut TokenStream<'a>,
    vars: &mut HashMap<&'a str, Var>,
) -> Result<(), TokenError> {
    use BindingType::*;
    #[derive(Copy, Clone)]
    enum BindingType {
        Global,
        Local,
        Block,
        BlockSize,
    }

    let typ = match tokens.next() {
        "global" => Global,
        "local" => Local,
        "block" => Block,
        "blocksize" => BlockSize,
        _ => return Err(tokens.prev_error()),
    };

    let c = tokens.next();
    if is_ident(c) {
        let key = c;
        let value = match typ {
            Global => Var::GlobalIndex(0),
            Local => Var::LocalIndex(0),
            Block => Var::BlockIndex(0),
            BlockSize => Var::BlockSize(0),
        };

        if key != "_" {
            vars.insert(key, value);
        }
    } else if c == "[" {
        for axis in 0.. {
            if axis as usize > MAX_DIMS {
                return Err(tokens.prev_error())?;
            }

            let key = tokens.next_if(is_ident)?;
            let value = match typ {
                Global => Var::GlobalIndex(axis),
                Local => Var::LocalIndex(axis),
                Block => Var::BlockIndex(axis),
                BlockSize => Var::BlockSize(axis),
            };

            if key != "_" {
                vars.insert(key, value);
            }

            match tokens.next() {
                "]" => break,
                "," => continue,
                _ => return Err(tokens.prev_error()),
            }
        }
    } else {
        return Err(tokens.prev_error());
    }

    Ok(())
}

fn parse_bindings<'a>(tokens: &mut TokenStream<'a>) -> Result<HashMap<&'a str, Var>, TokenError> {
    let mut vars = HashMap::default();

    if tokens.try_advance(|t| parse_binding(t, &mut vars)).is_ok() {
        while tokens.consume(",").is_ok() {
            parse_binding(tokens, &mut vars)?;
        }
    }

    Ok(vars)
}

fn parse_primitive(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<Expr, TokenError> {
    let c = tokens.next();

    let result = if is_ident(c) {
        match vars.get(c) {
            Some(var) => Expr::Var(*var),
            None => Expr::Arg(c.into()),
        }
    } else if let Some(number) = parse_number(c) {
        Expr::Immediate(number)
    } else if c == "-" {
        let prim = parse_primitive(tokens, vars)?;
        Expr::Mul(Box::new([prim, Expr::Immediate(-1)]))
    } else if c == "(" {
        let result = parse_expr(tokens, vars)?;
        tokens.consume(")")?;
        result
    } else {
        return Err(tokens.prev_error());
    };

    Ok(result)
}

fn parse_multiplication(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<Expr, TokenError> {
    let mut result = parse_primitive(tokens, vars)?;

    while let Ok(operator) = tokens.next_if(|c| c == "*" || c == "/") {
        let rhs = parse_primitive(tokens, vars)?;
        let operands = Box::new([result, rhs]);

        result = match operator {
            "*" => Expr::Mul(operands),
            "/" => Expr::Div(operands),
            _ => unreachable!(),
        };
    }

    Ok(result)
}

fn parse_addition(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<Expr, TokenError> {
    let mut result = parse_multiplication(tokens, vars)?;

    while let Ok(operator) = tokens.next_if(|c| c == "+" || c == "-") {
        let mut rhs = parse_multiplication(tokens, vars)?;

        if operator == "-" {
            rhs = Expr::Mul(Box::new([rhs, Expr::Immediate(-1)]));
        }

        result = Expr::Add(Box::new([result, rhs]));
    }

    Ok(result)
}

fn parse_expr(tokens: &mut TokenStream<'_>, vars: &HashMap<&str, Var>) -> Result<Expr, TokenError> {
    parse_addition(tokens, vars)
}

fn parse_slice(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<SliceExpr, TokenError> {
    let start = tokens
        .try_advance(|t| parse_expr(t, vars))
        .ok()
        .map(Box::new);

    if tokens.consume(":").is_ok() {
        let end = tokens
            .try_advance(|t| parse_expr(t, vars))
            .ok()
            .map(Box::new);

        let step = if tokens.consume(":").is_ok() {
            tokens
                .try_advance(|t| parse_expr(t, vars))
                .ok()
                .map(Box::new)
        } else {
            None
        };

        Ok(SliceExpr::Range { start, end, step })
    } else if let Some(index) = start {
        Ok(SliceExpr::Index { index })
    } else {
        Err(tokens.error())
    }
}

fn parse_access_mode(tokens: &mut TokenStream<'_>) -> Result<AccessMode, TokenError> {
    Ok(match tokens.next() {
        "read" => AccessMode::Read,
        "write" => AccessMode::Write,
        "readwrite" => AccessMode::ReadWrite,
        name @ ("reduce" | "atomic") => {
            tokens.consume("(")?;
            let reduction = match tokens.next() {
                "sum" | "+" => ReductionFunction::Sum,
                "and" | "&" => ReductionFunction::And,
                "or" | "|" => ReductionFunction::Or,
                "min" => ReductionFunction::Min,
                "max" => ReductionFunction::Max,
                _ => return Err(tokens.prev_error()),
            };
            tokens.consume(")")?;

            match name {
                "reduce" => AccessMode::Reduce(reduction),
                "atomic" => AccessMode::Atomic(reduction),
                _ => unreachable!(),
            }
        }
        _ => return Err(tokens.prev_error()),
    })
}

fn parse_argument(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<AnnotationRule, TokenError> {
    let access_mode = parse_access_mode(tokens)?;

    let name = tokens.next_if(is_ident)?;
    let mut slices = Vec::with_capacity(MAX_DIMS);

    if tokens.consume("[").is_ok() {
        loop {
            slices.push(parse_slice(tokens, vars)?);

            // If we reach MAX_DIMS, the next token MUST be ']'
            if slices.len() >= MAX_DIMS {
                tokens.consume("]")?;
                break;
            }

            // Next token can be:
            //   ',': continue
            //   '][': continue
            //   ']': break
            match tokens.next() {
                "," => continue,
                "]" => match tokens.consume("[") {
                    Ok(_) => continue,
                    Err(_) => break,
                },
                _ => return Err(tokens.prev_error()),
            }
        }
    }

    Ok(AnnotationRule {
        name: name.to_string(),
        access_mode,
        access_pattern: AccessPattern(slices.into_boxed_slice()),
    })
}

fn parse_arguments(
    tokens: &mut TokenStream<'_>,
    vars: &HashMap<&str, Var>,
) -> Result<Vec<AnnotationRule>, TokenError> {
    let mut rules = vec![];

    while !tokens.end() {
        rules.push(parse_argument(tokens, vars)?);

        if tokens.consume(",").is_err() {
            break;
        }
    }

    Ok(rules)
}

fn parse_toplevel(tokens: &mut TokenStream<'_>) -> Result<Vec<AnnotationRule>, TokenError> {
    let vars = parse_bindings(tokens)?;

    tokens.consume("=")?;
    tokens.consume(">")?;

    let result = parse_arguments(tokens, &vars)?;

    if !tokens.end() {
        return Err(tokens.error());
    }

    Ok(result)
}

#[derive(Debug)]
pub(crate) struct ParseError {
    input: String,
    span: Range<usize>,
}

impl Error for ParseError {}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (start, end) = (self.span.start, self.span.end);

        if start >= self.input.len() {
            write!(f, "error while parsing expression: unexpected end of line")
        } else {
            write!(
                f,
                "error while parsing expression: unexpected token {:?} at location {}:{}",
                &self.input[start..end],
                start,
                end - 1,
            )
        }
    }
}

pub(crate) fn parse_rules(input: &str) -> Result<Vec<AnnotationRule>, ParseError> {
    let n = input.len();
    let spans = tokenize(input);
    let mut stream = TokenStream::new(input, &spans);

    parse_toplevel(&mut stream).map_err(|TokenError { index }| ParseError {
        input: input.to_string(),
        span: spans.get(index).cloned().unwrap_or(n..n),
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let input = "123 + read A[i, 456]";
        let tokens = tokenize(input);
        let mut tokens = TokenStream::new(input, &tokens);
        assert_eq!(tokens.next(), "123");
        assert_eq!(tokens.next(), "+");
        assert_eq!(tokens.next(), "read");
        assert_eq!(tokens.next(), "A");
        assert_eq!(tokens.next(), "[");
        assert_eq!(tokens.next(), "i");
        assert_eq!(tokens.next(), ",");
        assert_eq!(tokens.next(), "456");
        assert_eq!(tokens.next(), "]");
        assert_eq!(tokens.next(), "");
        assert!(tokens.end());
    }

    #[test]
    fn test_slices() {
        fn f(i: bool, j: bool, k: bool) -> SliceExpr {
            let start = match i {
                true => Some(Box::new(Expr::Var(Var::GlobalIndex(0)))),
                false => None,
            };

            let end = match j {
                true => Some(Box::new(Expr::Var(Var::GlobalIndex(1)))),
                false => None,
            };

            let step = match k {
                true => Some(Box::new(Expr::Var(Var::GlobalIndex(2)))),
                false => None,
            };

            SliceExpr::Range { start, end, step }
        }
        let expected = [
            f(true, true, true),
            f(true, true, false),
            f(true, false, true),
            f(false, true, true),
            f(true, false, false),
            f(false, true, false),
            f(false, false, true),
            f(false, false, false),
            f(true, true, false),
            f(true, false, false),
            f(false, true, false),
            f(false, false, false),
        ];

        let input = "global [i,j,k] => \
        read A[i:j:k], \
        read A[i:j:], \
        read A[i::k], \
        read A[:j:k], \
        read A[i::], \
        read A[:j:], \
        read A[::k], \
        read A[::], \
        read A[i:j], \
        read A[i:], \
        read A[:j], \
        read A[:]";

        let rules = parse_rules(input).unwrap();
        assert_eq!(rules.len(), expected.len());

        for (rule, expect) in zip(&rules, &expected) {
            assert_eq!(rule.name, "A");
            assert_eq!(rule.access_mode, AccessMode::Read);
            assert_eq!(rule.access_pattern.len(), 1);

            let slice = &(rule.access_pattern.0)[0];

            assert_eq!(slice, expect);
        }
    }

    #[test]
    fn test_expression() {
        fn pattern<const N: usize>(slices: [SliceExpr; N]) -> AccessPattern {
            AccessPattern(Box::from(slices))
        }

        fn var(v: Var) -> Box<Expr> {
            Box::new(Expr::Var(v))
        }

        fn imm(c: i64) -> Box<Expr> {
            Box::new(Expr::Immediate(c))
        }

        let result = parse_rules(
            "global i, local j, block [k, m] =>
            read A[i],
            reduce(+) B[0],
            write C[i:i+1:2, :, i],
            readwrite D[i,j,k:m]",
        )
        .unwrap();

        let rule = AnnotationRule {
            name: "A".to_string(),
            access_mode: AccessMode::Read,
            access_pattern: pattern([SliceExpr::Index {
                index: var(Var::GlobalIndex(0)),
            }]),
        };

        assert_eq!(result[0], rule);

        let rule = AnnotationRule {
            name: "B".to_string(),
            access_mode: AccessMode::Reduce(ReductionFunction::Sum),
            access_pattern: pattern([SliceExpr::Index { index: imm(0) }]),
        };

        assert_eq!(result[1], rule);

        let rule = AnnotationRule {
            name: "C".to_string(),
            access_mode: AccessMode::Write,
            access_pattern: pattern([
                SliceExpr::Range {
                    start: Some(var(Var::GlobalIndex(0))),
                    end: Some(Box::new(Expr::Add(Box::from([
                        *var(Var::GlobalIndex(0)),
                        *imm(1),
                    ])))),
                    step: Some(imm(2)),
                },
                SliceExpr::Range {
                    start: None,
                    end: None,
                    step: None,
                },
                SliceExpr::Index {
                    index: var(Var::GlobalIndex(0)),
                },
            ]),
        };

        assert_eq!(result[2], rule);

        let rule = AnnotationRule {
            name: "D".to_string(),
            access_mode: AccessMode::ReadWrite,
            access_pattern: pattern([
                SliceExpr::Index {
                    index: var(Var::GlobalIndex(0)),
                },
                SliceExpr::Index {
                    index: var(Var::LocalIndex(0)),
                },
                SliceExpr::Range {
                    start: Some(var(Var::BlockIndex(0))),
                    end: Some(var(Var::BlockIndex(1))),
                    step: None,
                },
            ]),
        };

        assert_eq!(result[3], rule);
    }
}
