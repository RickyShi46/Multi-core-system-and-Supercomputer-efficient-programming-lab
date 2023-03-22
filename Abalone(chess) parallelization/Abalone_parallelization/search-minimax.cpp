/**
 * Minimax strategy:
 *
 *
 *
 * (c) 2022, Bat-Amgalan Bat-Erdene, Yu Shi, Jorge Padilla Perez (Group 3)
 */

#include "search.h"
#include "board.h"
#include "eval.h"
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
#include <cstring>

/**
 * To create your own search strategy:
 * - copy this file into another one,
 * - change the class name one the name given in constructor,
 * - adjust clone() to return an instance of your class
 * - adjust last line of this file to create a global instance
 *   of your class
 * - adjust the Makefile to include your class in SEARCH_OBJS
 * - implement searchBestMove()
 *
 * Advises for implementation of searchBestMove():
 * - call foundBestMove() when finding a best move since search start
 * - call finishedNode() when finishing evaluation of a tree node
 * - Use _maxDepth for strength level (maximal level searched in tree)
 */
class MinimaxStrategy : public SearchStrategy
{
public:
    // Defines the name of the strategy
    MinimaxStrategy() : SearchStrategy("Minimax") {}

    // Factory method: just return a new instance of this class
    SearchStrategy *clone() { return new MinimaxStrategy(); }

private:
    /**
     * Implementation of the strategy.
     */
    void searchBestMove();
    /* recursive minimax search top layer*/
    int minimaxPar(char depth, Board tempBoard, int& numberOfEval);
    /* recursive minimax search */
    int minimaxSeq(char depth, Board * tempBoard, Evaluator * ev, int alpha, int beta, int& numberOfEval);
    //check if same fields
    bool isSameFields(int* field1, int* field2);

    //last best Evaluation
    int _lastBestEval;

    /* prinicipal variation found in last search */
    Variation _pv;

    //total elapsed time in seconds
    double _remainingTime{60};
    //adaptiveDepth depending on time and moveNumber
    char _adaptiveDepth{5};
    //ownMoveNumber that counts only the moves it played, opponent moves not inclusive
    short _ownMoveNumber{1};
    //2move prior board field array
    int _field2Prior[121];
    //1move prior board field array
    int _field1Prior[121];
    //2move prior move
    Move _move2Prior;
    //1move prior move
    Move _move1Prior;
};

bool MinimaxStrategy::isSameFields(int* field1, int field2[121])
{
    for(int i = 0; i < 121; i++) {
        if ( field1[i] != field2[i] ) return false;
    }

    return true;
}


void MinimaxStrategy::searchBestMove()
{
    struct timeval t1, t2;
    int numberOfEval = 0;

    gettimeofday(&t1, 0);
    //if this board has occured throw the same move again
    if(_ownMoveNumber >= 3 && isSameFields(_board->fieldArray(), _field2Prior)){
        //throw the same move again
        _bestMove = _move2Prior;
        printf("Thrown prior move\n");
    }
    else{ //do minimax

        if(_remainingTime < 4) _adaptiveDepth = 3;
        else if(_remainingTime < 10) _adaptiveDepth = 4;
        else if(_lastBestEval < -900) _adaptiveDepth = 6;
        else if(_remainingTime > 20 && _lastBestEval < -400) _adaptiveDepth = 6;
        else _adaptiveDepth = 5;
        //_adaptiveDepth = _maxDepth;

        // main minimax calculations
        omp_set_dynamic(0);
        omp_set_num_threads(48);
        _lastBestEval = minimaxPar(0, *_board, numberOfEval); // depth start at 0 and goes till _adaptiveDepth (_adaptiveDepth is the depth of the leaf nodes)

        printf("final best Eval = %d\n", _lastBestEval);
        printf("Number of Evaluations = %d\n", numberOfEval);
    }
    gettimeofday(&t2, 0);

    double usecsPassed =
        (1000000.0 * t2.tv_sec + t2.tv_usec) -
        (1000000.0 * t1.tv_sec + t1.tv_usec);

    // printf("Microseconds passed = %f\n", usecsPassed);
    printf("Evaluations per second = %f * 10^6\n", numberOfEval / usecsPassed);
    _remainingTime -= (usecsPassed/1000000.0 + 0.02); //0.02s for safety
    printf("AdaptDepth = %d, Remain time = %fs, OwnMoveNumber = %d\n", (int)_adaptiveDepth, _remainingTime, _ownMoveNumber);

    //save the last 2 board positions and corresponding moves for rapid deployment of prior move
    if(_ownMoveNumber >= 2){
        std::memcpy(_field2Prior, _field1Prior, sizeof(_field1Prior));
        std::memcpy(_field1Prior, _board->fieldArray(), sizeof(_field1Prior));
        _move2Prior = _move1Prior;
        _move1Prior = _bestMove;
    }

    _ownMoveNumber++;
}

int MinimaxStrategy::minimaxPar(char depth, Board tempBoard, int& numberOfEval)
{
    bool maximizeTurn = true; //1st move is always our move, so we maximize it

    // we try to maximize bestEvaluation
    int bestEval = -35000;

    MoveList list;
    Move moves[150];

    // generate list of allowed moves, put them into <list>
    tempBoard.generateMoves(list);
    int nMoves = list.getLength();
    for(int i=0; i<nMoves; i++){
        list.getNext(moves[i]);
    }

    // loop over all moves
    #pragma omp parallel for schedule(dynamic,1) reduction(+: numberOfEval) shared(bestEval) firstprivate(tempBoard)
    for(int i=0; i<nMoves; i++)
    {
        Move m = moves[i];
        int eval;
        Evaluator ev;
        
        // draw move, evaluate, and restore position
        tempBoard.playMove(m);
        eval = minimaxSeq(depth + 1, &tempBoard, &ev, -35000, 35000, numberOfEval);
        tempBoard.takeBack();
        if (eval > bestEval)
        {
            #pragma omp critical
            {
            bestEval = eval;
            _bestMove = m;
            }
        }
    }

    // printf("best Eval = %d\n", bestEval);
    return bestEval;
}

int MinimaxStrategy::minimaxSeq(char depth, Board* tempBoard, Evaluator* ev, int alpha, int beta, int& numberOfEval)
{
    bool maximizeTurn = !(depth % 2); // even depth is maximizing, odd depth is minimizing

    if (depth >= _adaptiveDepth) //if leaf node is reached, evaluate the board
    {
        int eval = (1-maximizeTurn*2)*ev->calcEvaluation(tempBoard);
        //printf("nEval = %d, eval (leaf node)= %d, move = %s\n", _numberOfEval, eval, m.name());
        numberOfEval++;
        return eval;
    }

    // we try to maximize bestEvaluation
    int eval;

    MoveList list;
    Move m;

    // generate list of allowed moves, put them into <list>
    tempBoard->generateMoves(list);

    if(maximizeTurn){
        int bestValue = -35000; //initialize with the worst value
        // loop over all moves
        while(list.getNext(m))
        {
            // draw move, evaluate, and restore position
            tempBoard->playMove(m);
            eval = minimaxSeq(depth + 1, tempBoard, ev, alpha, beta, numberOfEval);
            tempBoard->takeBack();
            if(eval>bestValue){
                bestValue = eval;
            }
            if(bestValue>=beta){
                break;
            }
            if(alpha<eval){
                alpha = eval;
            }
        }
        return bestValue;
    }
    else{
        int worstValue = 35000; //initialize with the best value
        // loop over all moves
        while(list.getNext(m))
        {
            // draw move, evaluate, and restore position
            tempBoard->playMove(m);
            eval = minimaxSeq(depth + 1, tempBoard, ev, alpha, beta, numberOfEval);
            tempBoard->takeBack();
            if(eval<worstValue){
                worstValue = eval;
            }
            if(worstValue<=alpha){
                break;
            }
            if(beta>eval){
                beta = eval;
            }
        }
        return worstValue;
    }
}


// register ourselve as a search strategy
MinimaxStrategy minimaxStrategy;